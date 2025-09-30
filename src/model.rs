use anyhow::{Context, Result, anyhow};
use cust::device::Device;
use cust::memory::DeviceBuffer;
use cust::module::Module;
use cust::prelude::*;
use cust::stream::{Stream, StreamFlags};
use std::convert::TryFrom;

use crate::layers::{ActivationKind, FeedForwardNetwork, Layer, LayerNorm, MultiHeadAttention};
use crate::tensor::Tensor;

pub struct Embedding {
    weight: Tensor,
    vocab_size: usize,
    embed_dim: usize,
}

impl Embedding {
    pub fn new(device: &Device, vocab_size: usize, embed_dim: usize) -> Result<Self> {
        if vocab_size == 0 {
            return Err(anyhow!("vocab_size must be greater than zero"));
        }
        if embed_dim == 0 {
            return Err(anyhow!("embed_dim must be greater than zero"));
        }

        let weight = Tensor::randn(vec![vocab_size, embed_dim], device)?;
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

    pub fn forward(&self, indices: &[u32], batch_size: usize, seq_len: usize) -> Result<Tensor> {
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

        self.forward_i32(&device_indices, batch_size, seq_len)
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
        let module = Module::from_ptx(include_str!("embedding_lookup_kernel.ptx"), &[])
            .context("PTX load failed for embedding_lookup")?;
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
}

pub struct TransformerBlock {
    attention: MultiHeadAttention,
    norm_1: LayerNorm,
    norm_2: LayerNorm,
    feed_forward: FeedForwardNetwork,
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
}

impl Layer for TransformerBlock {
    fn forward(&self, input: &Tensor) -> Result<Tensor> {
        let normalized = self.norm_1.forward(input)?;
        let attention_output = self.attention.forward(&normalized)?;
        let residual_1 = input.add(&attention_output)?;

        let normalized_2 = self.norm_2.forward(&residual_1)?;
        let feed_forward_output = self.feed_forward.forward(&normalized_2)?;

        residual_1.add(&feed_forward_output)
    }
}

#[derive(Clone, Debug)]
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
        let lm_head_weight = Tensor::randn(vec![config.embed_dim, config.vocab_size], device)?;
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
}
