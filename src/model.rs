use crate::kernel_cache;
use anyhow::{Context, Result, anyhow};
use cust::device::Device;
use cust::memory::DeviceBuffer;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use serde::{Deserialize, Serialize};
use serde_json::{Map as JsonMap, Number as JsonNumber, Value as JsonValue};
use std::convert::TryFrom;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

use crate::layers::{
    ActivationKind, FeedForwardCache, FeedForwardGradients, FeedForwardNetwork, Layer, LayerNorm,
    LayerNormCache, LayerNormGradients, MultiHeadAttention, MultiHeadAttentionCache,
    MultiHeadAttentionGradients,
};
use crate::tensor::Tensor;

#[derive(Clone)]
pub struct Embedding {
    weight: Tensor,
    vocab_size: usize,
    embed_dim: usize,
}

#[derive(Clone)]
pub struct EmbeddingCache {
    indices: Vec<i32>,
    batch_size: usize,
    seq_len: usize,
}

impl Embedding {
    pub fn new(device: &Device, vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(anyhow!("vocab_size must be greater than zero"));
        }
        if embed_dim == 0 {
            return Err(anyhow!("embed_dim must be greater than zero"));
        }

        let scale = 1.0f32 / (embed_dim as f32).sqrt();
        let weight = Tensor::randn(vec![vocab_size, embed_dim], device)?;
        let weight = weight.mul_scalar(scale)?;
        Self::assemble(weight)
    }

    pub fn from_host(
        device: &Device,
        vocab_size: usize,
        embed_dim: usize,
        weights: Vec<f32>,
    ) -> Result<Self> {
        if vocab_size == 0 {
            return Err(anyhow!("vocab_size must be greater than zero"));
        }
        if embed_dim == 0 {
            return Err(anyhow!("embed_dim must be greater than zero"));
        }

        let weight = Tensor::from_host(weights, vec![vocab_size, embed_dim], device)?;
        Self::assemble(weight)
    }

    fn assemble(weight: Tensor) -> Result<Self> {
        if weight.shape().len() != 2 {
            return Err(anyhow!(
                "Embedding weights must be a 2D matrix, got shape {:?}",
                weight.shape()
            ));
        }

        let vocab_size = weight.shape()[0];
        let embed_dim = weight.shape()[1];

        Ok(Self {
            weight,
            vocab_size,
            embed_dim,
        })
    }

    pub fn weight(&self) -> &Tensor {
        &self.weight
    }

    pub fn weight_mut(&mut self) -> &mut Tensor {
        &mut self.weight
    }

    pub fn visit_parameters_mut<F>(&mut self, prefix: &str, f: &mut F) -> Result<()>
    where
        F: FnMut(&str, &mut Tensor) -> Result<()>,
    {
        f(&format!("{prefix}.weight"), &mut self.weight)?;
        Ok(())
    }

    pub fn forward(&self, indices: &[u32], batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let (output, _) = self.forward_with_cache(indices, batch_size, seq_len)?;
        Ok(output)
    }

    pub fn forward_with_cache(
        &self,
        indices: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, EmbeddingCache)> {
        if batch_size == 0 {
            return Err(anyhow!("Embedding forward expects batch_size > 0"));
        }
        if seq_len == 0 {
            return Err(anyhow!("Embedding forward expects seq_len > 0"));
        }

        let expected = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| anyhow!("Embedding forward dimensions overflow"))?;
        if indices.len() != expected {
            return Err(anyhow!(
                "Embedding expected {} indices (batch_size * seq_len) but received {}",
                expected,
                indices.len()
            ));
        }

        let mut device_indices = Vec::with_capacity(indices.len());
        for (position, &index) in indices.iter().enumerate() {
            if (index as usize) >= self.vocab_size {
                return Err(anyhow!(
                    "Token index {} at position {} exceeds vocab size {}",
                    index,
                    position,
                    self.vocab_size
                ));
            }
            let index_i32 = i32::try_from(index)
                .map_err(|_| anyhow!("Token index {} cannot be represented as i32", index))?;
            device_indices.push(index_i32);
        }

        let output = self.forward_i32(&device_indices, batch_size, seq_len)?;
        let cache = EmbeddingCache {
            indices: device_indices,
            batch_size,
            seq_len,
        };
        Ok((output, cache))
    }

    fn forward_i32(&self, indices: &[i32], batch_size: usize, seq_len: usize) -> Result<Tensor> {
        let embed_dim_i32 = i32::try_from(self.embed_dim)
            .map_err(|_| anyhow!("embed_dim {} too large for kernel launch", self.embed_dim))?;
        let seq_len_i32 = i32::try_from(seq_len)
            .map_err(|_| anyhow!("seq_len {} too large for kernel launch", seq_len))?;
        let batch_size_i32 = i32::try_from(batch_size)
            .map_err(|_| anyhow!("batch_size {} too large for kernel launch", batch_size))?;

        let total_elems = batch_size
            .checked_mul(seq_len)
            .and_then(|v| v.checked_mul(self.embed_dim))
            .ok_or_else(|| anyhow!("Embedding launch configuration overflow"))?;
        let total_u32 = u32::try_from(total_elems)
            .map_err(|_| anyhow!("Embedding launch exceeds CUDA grid limits"))?;

        let indices_buffer = DeviceBuffer::from_slice(indices)?;
        let result = Tensor::new(
            vec![batch_size, seq_len, self.embed_dim],
            &self.weight.device,
        )?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;
        let module = kernel_cache::module(
            include_str!("embedding_lookup_kernel.ptx"),
            "embedding_lookup",
        )?;
        let function = module
            .get_function("embedding_lookup_kernel")
            .context("Kernel load failed for embedding_lookup")?;

        let block_size = 256u32;
        let grid_size = (total_u32 + block_size - 1) / block_size;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                self.weight.data.as_device_ptr(),
                indices_buffer.as_device_ptr(),
                result.data.as_device_ptr(),
                embed_dim_i32,
                seq_len_i32,
                batch_size_i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for embedding_lookup")?;

        Ok(result)
    }

    pub fn backward(&self, cache: &EmbeddingCache, grad_output: &Tensor) -> Result<Tensor> {
        if grad_output.shape().len() != 3
            || grad_output.shape()[0] != cache.batch_size
            || grad_output.shape()[1] != cache.seq_len
            || grad_output.shape()[2] != self.embed_dim
        {
            return Err(anyhow!(
                "Embedding backward expects grad_output shape [{}, {}, {}], got {:?}",
                cache.batch_size,
                cache.seq_len,
                self.embed_dim,
                grad_output.shape()
            ));
        }

        if grad_output.device.as_raw() != self.weight.device.as_raw() {
            return Err(anyhow!(
                "Embedding grad_output must reside on the same device as the weights"
            ));
        }

        let total = cache
            .batch_size
            .checked_mul(cache.seq_len)
            .and_then(|v| v.checked_mul(self.embed_dim))
            .ok_or_else(|| anyhow!("Embedding backward launch configuration overflow"))?;
        if total > i32::MAX as usize {
            return Err(anyhow!(
                "Embedding backward tensor too large for kernel launch"
            ));
        }

        let grad_weight = Tensor::new(vec![self.vocab_size, self.embed_dim], &self.weight.device)?;
        let indices_buffer = DeviceBuffer::from_slice(&cache.indices)?;

        let block_size = 256u32;
        let grid_size = ((total as u32) + block_size - 1) / block_size;

        let module = kernel_cache::module(
            include_str!("embedding_backward_kernel.ptx"),
            "embedding_backward",
        )?;
        let function = module
            .get_function("embedding_backward_kernel")
            .context("Kernel load failed for embedding_backward_kernel")?;
        let stream =
            Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

        unsafe {
            launch!(function<<<grid_size, block_size, 0, stream>>>(
                grad_output.data.as_device_ptr(),
                indices_buffer.as_device_ptr(),
                grad_weight.data.as_device_ptr(),
                cache.batch_size as i32,
                cache.seq_len as i32,
                self.embed_dim as i32
            ))?;
        }

        stream
            .synchronize()
            .context("Stream sync failed for embedding_backward_kernel")?;

        Ok(grad_weight)
    }
}

pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm_1: LayerNorm,
    norm_2: LayerNorm,
    feed_forward: FeedForwardNetwork,
}

pub struct TransformerBlockCache {
    input_shape: Vec<usize>,
    norm1: LayerNormCache,
    attention: MultiHeadAttentionCache,
    norm2: LayerNormCache,
    feed_forward: FeedForwardCache,
}

pub struct TransformerBlockGradients {
    pub attention: MultiHeadAttentionGradients,
    pub norm1: LayerNormGradients,
    pub norm2: LayerNormGradients,
    pub feed_forward: FeedForwardGradients,
}

impl TransformerBlock {
    pub fn new(
        device: &Device,
        embed_dim: usize,
        num_heads: usize,
        feed_forward_dim: usize,
        activation: ActivationKind,
        layer_norm_eps: f32,
    ) -> Result<Self> {
        if layer_norm_eps <= 0.0 {
            return Err(anyhow!("layer_norm_eps must be positive"));
        }

        let attention = MultiHeadAttention::new(device, embed_dim, num_heads)?;
        let norm_1 = LayerNorm::new(device, embed_dim, layer_norm_eps)?;
        let norm_2 = LayerNorm::new(device, embed_dim, layer_norm_eps)?;
        let feed_forward =
            FeedForwardNetwork::new(device, embed_dim, feed_forward_dim, activation)?;

        Ok(Self {
            attention,
            norm_1,
            norm_2,
            feed_forward,
        })
    }

    pub fn forward_with_cache(&self, input: &Tensor) -> Result<(Tensor, TransformerBlockCache)> {
        let (normalized, norm1_cache) = self.norm_1.forward_with_cache(input)?;
        let (attention_output, attention_cache) = self.attention.forward_with_cache(&normalized)?;
        let residual_1 = input.add(&attention_output)?;

        let (normalized_2, norm2_cache) = self.norm_2.forward_with_cache(&residual_1)?;
        let (feed_forward_output, feed_forward_cache) =
            self.feed_forward.forward_with_cache(&normalized_2)?;

        let output = residual_1.add(&feed_forward_output)?;

        let cache = TransformerBlockCache {
            input_shape: input.shape().to_vec(),
            norm1: norm1_cache,
            attention: attention_cache,
            norm2: norm2_cache,
            feed_forward: feed_forward_cache,
        };

        Ok((output, cache))
    }

    pub fn backward(
        &self,
        cache: &TransformerBlockCache,
        grad_output: &Tensor,
    ) -> Result<(Tensor, TransformerBlockGradients)> {
        if grad_output.shape() != cache.input_shape.as_slice() {
            return Err(anyhow!(
                "TransformerBlock backward expects grad_output with shape {:?}, got {:?}",
                cache.input_shape,
                grad_output.shape()
            ));
        }

        let (grad_norm2_input, feed_forward_grads) = self
            .feed_forward
            .backward(&cache.feed_forward, grad_output)?;

        let (grad_residual_from_norm2, norm2_grads) =
            self.norm_2.backward(&cache.norm2, &grad_norm2_input)?;

        let grad_residual_total = grad_residual_from_norm2.add(grad_output)?;

        let grad_input_from_residual = grad_residual_total.clone();
        let grad_attention_output = grad_residual_total;

        let (grad_norm1_output, attention_grads) = self
            .attention
            .backward(&cache.attention, &grad_attention_output)?;

        let (grad_input_from_norm1, norm1_grads) =
            self.norm_1.backward(&cache.norm1, &grad_norm1_output)?;

        let grad_input = grad_input_from_residual.add(&grad_input_from_norm1)?;

        let gradients = TransformerBlockGradients {
            attention: attention_grads,
            norm1: norm1_grads,
            norm2: norm2_grads,
            feed_forward: feed_forward_grads,
        };

        Ok((grad_input, gradients))
    }

    fn append_named_tensors<'a>(&'a self, index: usize, out: &mut Vec<(String, &'a Tensor)>) {
        let prefix = format!("blocks.{index}");
        self.attention
            .append_named_tensors(&format!("{prefix}.attention"), out);
        self.norm_1
            .append_named_tensors(&format!("{prefix}.norm_1"), out);
        self.norm_2
            .append_named_tensors(&format!("{prefix}.norm_2"), out);
        self.feed_forward
            .append_named_tensors(&format!("{prefix}.mlp"), out);
    }

    fn visit_parameters_mut<F>(&mut self, index: usize, f: &mut F) -> Result<()>
    where
        F: FnMut(&str, &mut Tensor) -> Result<()>,
    {
        let prefix = format!("blocks.{index}");
        self.attention
            .visit_parameters_mut(&format!("{prefix}.attention"), f)?;
        self.norm_1
            .visit_parameters_mut(&format!("{prefix}.norm_1"), f)?;
        self.norm_2
            .visit_parameters_mut(&format!("{prefix}.norm_2"), f)?;
        self.feed_forward
            .visit_parameters_mut(&format!("{prefix}.mlp"), f)?;
        Ok(())
    }
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let (output, _) = self.forward_with_cache(input)?;
        Ok(output)
    }
}

pub struct GPTForwardCache {
    batch_size: usize,
    seq_len: usize,
    token_embedding: EmbeddingCache,
    position_embedding: EmbeddingCache,
    block_caches: Vec<TransformerBlockCache>,
    final_norm: LayerNormCache,
}

pub struct GPTGradients {
    pub token_embedding: Tensor,
    pub position_embedding: Tensor,
    pub blocks: Vec<TransformerBlockGradients>,
    pub final_norm: LayerNormGradients,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GPTConfig {
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub embed_dim: usize,
    pub num_heads: usize,
    pub num_layers: usize,
    pub feed_forward_dim: usize,
    pub layer_norm_eps: f32,
    pub activation: ActivationKind,
}

pub struct GPTModel {
    config: GPTConfig,
    device: Device,
    token_embedding: Embedding,
    position_embedding: Embedding,
    blocks: Vec<TransformerBlock>,
    final_norm: LayerNorm,
    lm_head_weight: Tensor,
    lm_head_bias: Option<Tensor>,
}

impl GPTModel {
    pub fn new(device: &Device, config: GPTConfig) -> Result<Self> {
        if config.vocab_size == 0 {
            return Err(anyhow!("GPTConfig vocab_size must be greater than zero"));
        }
        if config.max_seq_len == 0 {
            return Err(anyhow!("GPTConfig max_seq_len must be greater than zero"));
        }
        if config.embed_dim == 0 {
            return Err(anyhow!("GPTConfig embed_dim must be greater than zero"));
        }
        if config.num_heads == 0 {
            return Err(anyhow!("GPTConfig num_heads must be greater than zero"));
        }
        if config.num_layers == 0 {
            return Err(anyhow!("GPTConfig num_layers must be greater than zero"));
        }
        if config.feed_forward_dim == 0 {
            return Err(anyhow!(
                "GPTConfig feed_forward_dim must be greater than zero"
            ));
        }
        if config.layer_norm_eps <= 0.0 {
            return Err(anyhow!("GPTConfig layer_norm_eps must be positive"));
        }

        let token_embedding = Embedding::new(device, config.vocab_size, config.embed_dim)?;
        let position_embedding = Embedding::new(device, config.max_seq_len, config.embed_dim)?;

        let mut blocks = Vec::with_capacity(config.num_layers);
        for _ in 0..config.num_layers {
            blocks.push(TransformerBlock::new(
                device,
                config.embed_dim,
                config.num_heads,
                config.feed_forward_dim,
                config.activation,
                config.layer_norm_eps,
            )?);
        }

        let final_norm = LayerNorm::new(device, config.embed_dim, config.layer_norm_eps)?;
        let scale = 1.0f32 / (config.embed_dim as f32).sqrt();
        let lm_head_weight = Tensor::randn(vec![config.embed_dim, config.vocab_size], device)?;
        let lm_head_weight = lm_head_weight.mul_scalar(scale)?;
        let lm_head_bias = Some(Tensor::new(vec![1, config.vocab_size], device)?);

        Ok(Self {
            config,
            device: device.clone(),
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
            lm_head_weight,
            lm_head_bias,
        })
    }

    pub fn save_checkpoint<P: AsRef<Path>>(&self, output_dir: P) -> Result<()> {
        let output_dir = output_dir.as_ref();
        fs::create_dir_all(output_dir).with_context(|| {
            format!("Failed to create checkpoint directory at {:?}", output_dir)
        })?;

        let config_path = output_dir.join("config.json");
        let config_file = File::create(&config_path)
            .with_context(|| format!("Failed to create {:?}", config_path))?;
        serde_json::to_writer_pretty(config_file, &self.config)
            .context("Failed to write checkpoint config")?;

        let mut tensors: Vec<(String, &Tensor)> = Vec::new();
        tensors.push((
            "token_embedding.weight".to_string(),
            self.token_embedding.weight(),
        ));
        tensors.push((
            "position_embedding.weight".to_string(),
            self.position_embedding.weight(),
        ));

        for (idx, block) in self.blocks.iter().enumerate() {
            block.append_named_tensors(idx, &mut tensors);
        }

        self.final_norm
            .append_named_tensors("final_norm", &mut tensors);
        tensors.push(("lm_head.weight".to_string(), &self.lm_head_weight));
        if let Some(bias) = &self.lm_head_bias {
            tensors.push(("lm_head.bias".to_string(), bias));
        }

        let safetensors_path = output_dir.join("model.safetensors");
        let mut header = JsonMap::new();
        let mut offset_bytes: u64 = 0;

        for (name, tensor) in &tensors {
            let num_elements = tensor.size() as u64;
            let byte_len = num_elements
                .checked_mul(4)
                .ok_or_else(|| anyhow!("Tensor {name} is too large for safetensors export"))?;

            let mut entry = JsonMap::new();
            entry.insert("dtype".to_string(), JsonValue::String("F32".to_string()));

            let shape_values = tensor
                .shape()
                .iter()
                .map(|&dim| JsonValue::Number(JsonNumber::from(dim as u64)))
                .collect();
            entry.insert("shape".to_string(), JsonValue::Array(shape_values));

            entry.insert(
                "data_offsets".to_string(),
                JsonValue::Array(vec![
                    JsonValue::Number(JsonNumber::from(offset_bytes)),
                    JsonValue::Number(JsonNumber::from(
                        offset_bytes
                            .checked_add(byte_len)
                            .ok_or_else(|| anyhow!("Tensor {name} offset overflow"))?,
                    )),
                ]),
            );

            header.insert(name.clone(), JsonValue::Object(entry));
            offset_bytes = offset_bytes
                .checked_add(byte_len)
                .ok_or_else(|| anyhow!("Checkpoint size overflow"))?;
        }

        let header_bytes = serde_json::to_vec(&JsonValue::Object(header))
            .context("Failed to encode safetensors header")?;

        let mut writer = BufWriter::new(
            File::create(&safetensors_path)
                .with_context(|| format!("Failed to create {:?}", safetensors_path))?,
        );

        let header_len = header_bytes.len() as u64;
        writer
            .write_all(&header_len.to_le_bytes())
            .context("Failed to write safetensors header length")?;
        writer
            .write_all(&header_bytes)
            .context("Failed to write safetensors header")?;

        for (name, tensor) in &tensors {
            let host_data = tensor
                .to_host()
                .with_context(|| format!("Failed to copy tensor {name} to host"))?;
            for value in host_data {
                writer
                    .write_all(&value.to_le_bytes())
                    .with_context(|| format!("Failed to write tensor data for {name}"))?;
            }
        }

        writer.flush().context("Failed to flush safetensors file")?;

        Ok(())
    }

    pub fn forward_with_cache(
        &self,
        token_indices: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor, GPTForwardCache)> {
        if batch_size == 0 {
            return Err(anyhow!("GPTModel forward expects batch_size > 0"));
        }
        if seq_len == 0 {
            return Err(anyhow!("GPTModel forward expects seq_len > 0"));
        }
        if seq_len > self.config.max_seq_len {
            return Err(anyhow!(
                "seq_len {} exceeds configured max_seq_len {}",
                seq_len,
                self.config.max_seq_len
            ));
        }

        let expected_tokens = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| anyhow!("GPTModel forward dimensions overflow"))?;
        if token_indices.len() != expected_tokens {
            return Err(anyhow!(
                "Expected {} token indices (batch_size * seq_len) but received {}",
                expected_tokens,
                token_indices.len()
            ));
        }

        let (token_embeddings, token_cache) =
            self.token_embedding
                .forward_with_cache(token_indices, batch_size, seq_len)?;

        let mut position_indices = Vec::with_capacity(expected_tokens);
        for _ in 0..batch_size {
            for position in 0..seq_len {
                position_indices.push(position as u32);
            }
        }
        let (position_embeddings, position_cache) =
            self.position_embedding
                .forward_with_cache(&position_indices, batch_size, seq_len)?;

        let mut hidden = token_embeddings.add(&position_embeddings)?;
        let mut block_caches = Vec::with_capacity(self.blocks.len());
        for block in &self.blocks {
            let (next_hidden, cache) = block.forward_with_cache(&hidden)?;
            block_caches.push(cache);
            hidden = next_hidden;
        }

        let (final_hidden, final_norm_cache) = self.final_norm.forward_with_cache(&hidden)?;
        let flat_hidden =
            final_hidden.reshape(vec![batch_size * seq_len, self.config.embed_dim])?;

        let mut logits = flat_hidden.matmul(&self.lm_head_weight)?;
        if let Some(bias) = &self.lm_head_bias {
            logits = logits.add(bias)?;
        }

        let cache = GPTForwardCache {
            batch_size,
            seq_len,
            token_embedding: token_cache,
            position_embedding: position_cache,
            block_caches,
            final_norm: final_norm_cache,
        };

        Ok((flat_hidden, logits, cache))
    }

    fn forward_internal(
        &self,
        token_indices: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        if batch_size == 0 {
            return Err(anyhow!("GPTModel forward expects batch_size > 0"));
        }
        if seq_len == 0 {
            return Err(anyhow!("GPTModel forward expects seq_len > 0"));
        }
        if seq_len > self.config.max_seq_len {
            return Err(anyhow!(
                "seq_len {} exceeds configured max_seq_len {}",
                seq_len,
                self.config.max_seq_len
            ));
        }

        let expected_tokens = batch_size
            .checked_mul(seq_len)
            .ok_or_else(|| anyhow!("GPTModel forward dimensions overflow"))?;
        if token_indices.len() != expected_tokens {
            return Err(anyhow!(
                "Expected {} token indices (batch_size * seq_len) but received {}",
                expected_tokens,
                token_indices.len()
            ));
        }

        let token_embeddings = self
            .token_embedding
            .forward(token_indices, batch_size, seq_len)?;

        let mut position_indices = Vec::with_capacity(expected_tokens);
        for _ in 0..batch_size {
            for position in 0..seq_len {
                position_indices.push(position as u32);
            }
        }
        let position_embeddings =
            self.position_embedding
                .forward(&position_indices, batch_size, seq_len)?;

        let mut hidden = token_embeddings.add(&position_embeddings)?;
        for block in &self.blocks {
            hidden = block.forward(&hidden)?;
        }

        let hidden = self.final_norm.forward(&hidden)?;
        let flat_hidden = hidden.reshape(vec![batch_size * seq_len, self.config.embed_dim])?;

        let mut logits = flat_hidden.matmul(&self.lm_head_weight)?;
        if let Some(bias) = &self.lm_head_bias {
            logits = logits.add(bias)?;
        }

        Ok((flat_hidden, logits))
    }

    pub fn forward_with_hidden(
        &self,
        token_indices: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<(Tensor, Tensor)> {
        self.forward_internal(token_indices, batch_size, seq_len)
    }

    pub fn forward(
        &self,
        token_indices: &[u32],
        batch_size: usize,
        seq_len: usize,
    ) -> Result<Tensor> {
        let (_, logits) = self.forward_internal(token_indices, batch_size, seq_len)?;
        logits.reshape(vec![batch_size, seq_len, self.config.vocab_size])
    }

    pub fn backward(
        &self,
        cache: &GPTForwardCache,
        grad_flat_hidden: &Tensor,
    ) -> Result<GPTGradients> {
        let expected_tokens = cache
            .batch_size
            .checked_mul(cache.seq_len)
            .ok_or_else(|| anyhow!("GPTModel backward dimensions overflow"))?;

        if grad_flat_hidden.shape().len() != 2 {
            return Err(anyhow!(
                "GPTModel backward expects grad_flat_hidden to be 2D, got {:?}",
                grad_flat_hidden.shape()
            ));
        }

        if grad_flat_hidden.shape()[0] != expected_tokens
            || grad_flat_hidden.shape()[1] != self.config.embed_dim
        {
            return Err(anyhow!(
                "GPTModel backward expects grad_flat_hidden shape [{}, {}], got {:?}",
                expected_tokens,
                self.config.embed_dim,
                grad_flat_hidden.shape()
            ));
        }

        if grad_flat_hidden.device.as_raw() != self.device.as_raw() {
            return Err(anyhow!(
                "grad_flat_hidden device does not match model device"
            ));
        }

        if cache.block_caches.len() != self.blocks.len() {
            return Err(anyhow!(
                "GPTForwardCache block count {} does not match model block count {}",
                cache.block_caches.len(),
                self.blocks.len()
            ));
        }

        let grad_hidden = grad_flat_hidden.clone().reshape(vec![
            cache.batch_size,
            cache.seq_len,
            self.config.embed_dim,
        ])?;

        let (mut grad_current, final_norm_grads) =
            self.final_norm.backward(&cache.final_norm, &grad_hidden)?;

        let mut block_grads_rev = Vec::with_capacity(self.blocks.len());
        for (block, block_cache) in self.blocks.iter().zip(cache.block_caches.iter()).rev() {
            let (grad_prev, gradients) = block.backward(block_cache, &grad_current)?;
            block_grads_rev.push(gradients);
            grad_current = grad_prev;
        }
        block_grads_rev.reverse();

        let grad_token_embedding = self
            .token_embedding
            .backward(&cache.token_embedding, &grad_current)?;
        let grad_position_embedding = self
            .position_embedding
            .backward(&cache.position_embedding, &grad_current)?;

        Ok(GPTGradients {
            token_embedding: grad_token_embedding,
            position_embedding: grad_position_embedding,
            blocks: block_grads_rev,
            final_norm: final_norm_grads,
        })
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn config(&self) -> &GPTConfig {
        &self.config
    }

    pub fn lm_head_params_mut(&mut self) -> (&mut Tensor, Option<&mut Tensor>) {
        let bias = self.lm_head_bias.as_mut().map(|bias| bias as &mut Tensor);
        (&mut self.lm_head_weight, bias)
    }

    pub fn visit_parameters_mut<F>(&mut self, mut f: F) -> Result<()>
    where
        F: FnMut(&str, &mut Tensor) -> Result<()>,
    {
        self.token_embedding
            .visit_parameters_mut("token_embedding", &mut f)?;
        self.position_embedding
            .visit_parameters_mut("position_embedding", &mut f)?;

        for (idx, block) in self.blocks.iter_mut().enumerate() {
            block.visit_parameters_mut(idx, &mut f)?;
        }

        self.final_norm.visit_parameters_mut("final_norm", &mut f)?;

        f("lm_head.weight", &mut self.lm_head_weight)?;
        if let Some(bias) = &mut self.lm_head_bias {
            f("lm_head.bias", bias)?;
        }

        Ok(())
    }
}
