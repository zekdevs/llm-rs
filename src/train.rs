use crate::kernel_cache;
use crate::layers::{FeedForwardGradients, LayerNormGradients, MultiHeadAttentionGradients};
use crate::model::{GPTGradients, GPTModel, TransformerBlockGradients};
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow};
use cust::prelude::*;
use cust::{
    device::Device,
    memory::DeviceBuffer,
    stream::{Stream, StreamFlags},
};
use rand::seq::SliceRandom;
use std::collections::HashMap;
use std::fs::File;
use std::io::{BufRead, BufReader};
use std::path::Path;

/// Configuration for the basic stochastic gradient descent fine-tuning loop.
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub batch_size: usize,
    pub seq_len: usize,
    pub learning_rate: f32,
    pub epochs: usize,
    pub max_sequences_per_epoch: Option<usize>,
    pub shuffle_windows: bool,
    pub momentum: f32,
    pub weight_decay: f32,
    pub log_every: usize,
    pub gradient_clip_norm: Option<f32>,
}

impl TrainingConfig {
    pub fn validate(&self) -> Result<()> {
        if self.batch_size == 0 {
            return Err(anyhow!("TrainingConfig batch_size must be > 0"));
        }
        if self.seq_len == 0 {
            return Err(anyhow!("TrainingConfig seq_len must be > 0"));
        }
        if !(self.learning_rate.is_finite()) || self.learning_rate <= 0.0 {
            return Err(anyhow!(
                "TrainingConfig learning_rate must be finite and positive"
            ));
        }
        if self.epochs == 0 {
            return Err(anyhow!("TrainingConfig epochs must be > 0"));
        }
        if !(self.momentum.is_finite()) || self.momentum < 0.0 || self.momentum >= 1.0 {
            return Err(anyhow!(
                "TrainingConfig momentum must be finite and in [0, 1)"
            ));
        }
        if !(self.weight_decay.is_finite()) || self.weight_decay < 0.0 {
            return Err(anyhow!(
                "TrainingConfig weight_decay must be finite and non-negative"
            ));
        }
        if self.log_every == 0 {
            return Err(anyhow!("TrainingConfig log_every must be > 0"));
        }
        if let Some(clip) = self.gradient_clip_norm {
            if !clip.is_finite() || clip <= 0.0 {
                return Err(anyhow!(
                    "TrainingConfig gradient_clip_norm must be finite and positive when set"
                ));
            }
        }
        Ok(())
    }
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            batch_size: 4,
            seq_len: 128,
            learning_rate: 1e-3,
            epochs: 1,
            max_sequences_per_epoch: Some(1024),
            shuffle_windows: true,
            momentum: 0.9,
            weight_decay: 1e-2,
            log_every: 100,
            gradient_clip_norm: Some(1.0),
        }
    }
}

/// Summary statistics returned by the training loop.
#[derive(Debug, Clone)]
pub struct TrainingReport {
    pub epoch_losses: Vec<f32>,
    pub batches_per_epoch: Vec<usize>,
    pub tokens_per_epoch: Vec<usize>,
    pub total_batches: usize,
    pub total_tokens: usize,
}

struct LmHeadOptimizer {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    embed_dim: usize,
    vocab_size: usize,
    velocity_weight: Tensor,
    velocity_bias: Option<Tensor>,
    gradient_clip_norm: Option<f32>,
}

impl LmHeadOptimizer {
    fn new(
        device: &Device,
        embed_dim: usize,
        vocab_size: usize,
        has_bias: bool,
        config: &TrainingConfig,
    ) -> Result<Self> {
        let velocity_weight = Tensor::new(vec![embed_dim, vocab_size], device)?;
        let velocity_bias = if has_bias {
            Some(Tensor::new(vec![1, vocab_size], device)?)
        } else {
            None
        };

        Ok(Self {
            learning_rate: config.learning_rate,
            momentum: config.momentum,
            weight_decay: config.weight_decay,
            embed_dim,
            vocab_size,
            velocity_weight,
            velocity_bias,
            gradient_clip_norm: config.gradient_clip_norm,
        })
    }

    fn step(
        &mut self,
        model: &mut GPTModel,
        hidden: Tensor,
        logits: Tensor,
        targets: &[u32],
    ) -> Result<(Tensor, f32)> {
        if hidden.shape().len() != 2 || logits.shape().len() != 2 {
            return Err(anyhow!("Expected 2D tensors for hidden states and logits"));
        }

        let batch_seq = hidden.shape()[0];
        let embed_dim = hidden.shape()[1];
        let logits_rows = logits.shape()[0];
        let vocab_size = logits.shape()[1];

        if logits_rows != batch_seq {
            return Err(anyhow!(
                "Hidden state count ({}) must match logits rows ({})",
                batch_seq,
                logits_rows
            ));
        }
        if embed_dim != self.embed_dim {
            return Err(anyhow!(
                "Hidden dimension {} does not match lm_head embed_dim {}",
                embed_dim,
                self.embed_dim
            ));
        }
        if vocab_size != self.vocab_size {
            return Err(anyhow!(
                "Logit vocab {} does not match lm_head vocab_size {}",
                vocab_size,
                self.vocab_size
            ));
        }
        if targets.len() != batch_seq {
            return Err(anyhow!(
                "Target length {} does not match batch*seq {}",
                targets.len(),
                batch_seq
            ));
        }

        let inv_batch = 1.0f32 / (batch_seq as f32);
        let probs = logits.softmax()?;
        let (grad_logits, loss) = cross_entropy_backward_with_loss(&probs, targets, inv_batch)?;

        let (weight_tensor, bias_tensor_opt) = model.lm_head_params_mut();
        let weight_snapshot = weight_tensor.clone();

        let (mut grad_hidden, mut grad_weight) =
            Tensor::matmul_backward(&hidden, &weight_snapshot, &grad_logits)?;

        let mut grad_bias = if bias_tensor_opt.is_some() {
            let grad_logits_t = grad_logits.transpose2d()?;
            Some(grad_logits_t.sum_rows()?.reshape(vec![1, vocab_size])?)
        } else {
            None
        };

        if let Some(clip_norm) = self.gradient_clip_norm {
            let mut total_sq = grad_weight.l2_norm_squared()?;
            if let Some(ref gb) = grad_bias {
                total_sq += gb.l2_norm_squared()?;
            }
            let total_norm = total_sq.sqrt();
            if total_norm > clip_norm {
                let scale = clip_norm / (total_norm + 1e-6);
                grad_weight = grad_weight.mul_scalar(scale)?;
                if let Some(ref mut gb) = grad_bias {
                    *gb = gb.mul_scalar(scale)?;
                }
                grad_hidden = grad_hidden.mul_scalar(scale)?;
            }
        }

        let weight_decay_term = weight_tensor.mul_scalar(self.weight_decay)?;
        grad_weight = grad_weight.add(&weight_decay_term)?;

        let momentum_velocity = self.velocity_weight.mul_scalar(self.momentum)?;
        self.velocity_weight = momentum_velocity.add(&grad_weight)?;
        let weight_update = self.velocity_weight.mul_scalar(self.learning_rate)?;
        let new_weight = weight_tensor.sub(&weight_update)?;
        *weight_tensor = new_weight;

        if let Some(bias_tensor) = bias_tensor_opt {
            if let Some(velocity_bias) = &mut self.velocity_bias {
                if let Some(grad_bias_tensor) = grad_bias {
                    let momentum_bias = velocity_bias.mul_scalar(self.momentum)?;
                    *velocity_bias = momentum_bias.add(&grad_bias_tensor)?;
                    let bias_update = velocity_bias.mul_scalar(self.learning_rate)?;
                    let new_bias = bias_tensor.sub(&bias_update)?;
                    *bias_tensor = new_bias;
                }
            }
        }

        Ok((grad_hidden, loss))
    }
}

struct EncoderOptimizer {
    learning_rate: f32,
    momentum: f32,
    weight_decay: f32,
    velocities: HashMap<String, Tensor>,
    gradient_clip_norm: Option<f32>,
}

impl EncoderOptimizer {
    fn new(model: &mut GPTModel, config: &TrainingConfig) -> Result<Self> {
        let mut velocities = HashMap::new();
        model.visit_parameters_mut(|name, param| {
            if name.starts_with("lm_head.") {
                return Ok(());
            }
            let velocity = Tensor::new(param.shape().to_vec(), &param.device)?;
            velocities.insert(name.to_string(), velocity);
            Ok(())
        })?;

        Ok(Self {
            learning_rate: config.learning_rate,
            momentum: config.momentum,
            weight_decay: config.weight_decay,
            velocities,
            gradient_clip_norm: config.gradient_clip_norm,
        })
    }

    fn step(&mut self, model: &mut GPTModel, grads: GPTGradients) -> Result<()> {
        let mut grad_map = Self::collect_gradients(grads);

        if let Some(clip_norm) = self.gradient_clip_norm {
            let mut total_sq = 0f32;
            for grad in grad_map.values() {
                total_sq += grad.l2_norm_squared()?;
            }
            let total_norm = total_sq.sqrt();
            if total_norm > clip_norm {
                let scale = clip_norm / (total_norm + 1e-6);
                for grad in grad_map.values_mut() {
                    *grad = grad.mul_scalar(scale)?;
                }
            }
        }

        model.visit_parameters_mut(|name, param| {
            if name.starts_with("lm_head.") {
                return Ok(());
            }

            let grad = grad_map
                .remove(name)
                .ok_or_else(|| anyhow!("Missing gradient for parameter {name}"))?;
            self.apply_update(name, param, grad)?;
            Ok(())
        })?;

        if !grad_map.is_empty() {
            let leftover: Vec<String> = grad_map.keys().cloned().collect();
            return Err(anyhow!(
                "Unused gradients provided for parameters: {:?}",
                leftover
            ));
        }

        Ok(())
    }

    fn apply_update(&mut self, name: &str, param: &mut Tensor, grad: Tensor) -> Result<()> {
        let velocity = self
            .velocities
            .get_mut(name)
            .ok_or_else(|| anyhow!("Missing velocity for parameter {name}"))?;

        let mut grad_accum = grad;
        if Self::should_apply_weight_decay(name) && self.weight_decay > 0.0 {
            let decay = param.mul_scalar(self.weight_decay)?;
            grad_accum = grad_accum.add(&decay)?;
        }

        let momentum_term = velocity.mul_scalar(self.momentum)?;
        *velocity = momentum_term.add(&grad_accum)?;
        let update = velocity.mul_scalar(self.learning_rate)?;
        let new_param = param.sub(&update)?;
        *param = new_param;
        Ok(())
    }

    fn collect_gradients(grads: GPTGradients) -> HashMap<String, Tensor> {
        let GPTGradients {
            token_embedding,
            position_embedding,
            blocks,
            final_norm,
        } = grads;

        let LayerNormGradients {
            gamma: final_gamma,
            beta: final_beta,
        } = final_norm;

        let mut map = HashMap::new();
        map.insert("token_embedding.weight".to_string(), token_embedding);
        map.insert("position_embedding.weight".to_string(), position_embedding);
        map.insert("final_norm.gamma".to_string(), final_gamma);
        map.insert("final_norm.beta".to_string(), final_beta);

        for (index, block_grad) in blocks.into_iter().enumerate() {
            let TransformerBlockGradients {
                attention,
                norm1,
                norm2,
                feed_forward,
            } = block_grad;

            let MultiHeadAttentionGradients {
                w_q,
                w_k,
                w_v,
                w_o,
                b_q,
                b_k,
                b_v,
                b_o,
            } = attention;

            let LayerNormGradients {
                gamma: norm1_gamma,
                beta: norm1_beta,
            } = norm1;
            let LayerNormGradients {
                gamma: norm2_gamma,
                beta: norm2_beta,
            } = norm2;

            let FeedForwardGradients { w1, b1, w2, b2 } = feed_forward;

            let prefix = format!("blocks.{index}");
            let attention_prefix = format!("{prefix}.attention");
            map.insert(format!("{attention_prefix}.w_q"), w_q);
            map.insert(format!("{attention_prefix}.w_k"), w_k);
            map.insert(format!("{attention_prefix}.w_v"), w_v);
            map.insert(format!("{attention_prefix}.w_o"), w_o);
            if let Some(b_q) = b_q {
                map.insert(format!("{attention_prefix}.b_q"), b_q);
            }
            if let Some(b_k) = b_k {
                map.insert(format!("{attention_prefix}.b_k"), b_k);
            }
            if let Some(b_v) = b_v {
                map.insert(format!("{attention_prefix}.b_v"), b_v);
            }
            if let Some(b_o) = b_o {
                map.insert(format!("{attention_prefix}.b_o"), b_o);
            }

            let norm1_prefix = format!("{prefix}.norm_1");
            map.insert(format!("{norm1_prefix}.gamma"), norm1_gamma);
            map.insert(format!("{norm1_prefix}.beta"), norm1_beta);

            let norm2_prefix = format!("{prefix}.norm_2");
            map.insert(format!("{norm2_prefix}.gamma"), norm2_gamma);
            map.insert(format!("{norm2_prefix}.beta"), norm2_beta);

            let mlp_prefix = format!("{prefix}.mlp");
            map.insert(format!("{mlp_prefix}.w1"), w1);
            map.insert(format!("{mlp_prefix}.w2"), w2);
            if let Some(b1) = b1 {
                map.insert(format!("{mlp_prefix}.b1"), b1);
            }
            if let Some(b2) = b2 {
                map.insert(format!("{mlp_prefix}.b2"), b2);
            }
        }

        map
    }

    fn should_apply_weight_decay(name: &str) -> bool {
        name.ends_with(".weight")
            || name.ends_with(".w_q")
            || name.ends_with(".w_k")
            || name.ends_with(".w_v")
            || name.ends_with(".w_o")
            || name.ends_with(".w1")
            || name.ends_with(".w2")
    }
}

fn cross_entropy_backward_with_loss(
    probs: &Tensor,
    targets: &[u32],
    inv_batch: f32,
) -> Result<(Tensor, f32)> {
    if probs.shape().len() != 2 {
        return Err(anyhow!(
            "cross_entropy_backward_with_loss expects a 2D tensor, got {:?}",
            probs.shape()
        ));
    }

    let batch_seq = probs.shape()[0];
    let vocab_size = probs.shape()[1];

    if targets.len() != batch_seq {
        return Err(anyhow!(
            "Target length {} does not match batch*seq {}",
            targets.len(),
            batch_seq
        ));
    }

    let targets_device = DeviceBuffer::from_slice(targets)
        .context("Failed to copy targets to device for cross-entropy")?;

    let module = kernel_cache::module(include_str!("cross_entropy_kernel.ptx"), "cross_entropy")?;
    let backward_function = module
        .get_function("cross_entropy_backward_kernel")
        .context("Kernel load failed for cross_entropy_backward_kernel")?;
    let gather_function = module
        .get_function("gather_target_prob_kernel")
        .context("Kernel load failed for gather_target_prob_kernel")?;

    let grad_logits = Tensor::new(vec![batch_seq, vocab_size], &probs.device)?;
    let target_probs = Tensor::new(vec![batch_seq], &probs.device)?;

    let block_size = 256u32;
    let grad_total = (batch_seq
        .checked_mul(vocab_size)
        .ok_or_else(|| anyhow!("cross-entropy tensor size overflow"))?) as u32;
    let grad_grid = (grad_total + block_size - 1) / block_size;
    let gather_grid = ((batch_seq as u32) + block_size - 1) / block_size;

    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

    unsafe {
        launch!(backward_function<<<grad_grid, block_size, 0, stream>>>(
            probs.data.as_device_ptr(),
            targets_device.as_device_ptr(),
            grad_logits.data.as_device_ptr(),
            inv_batch,
            vocab_size as i32,
            batch_seq as i32
        ))?;

        launch!(gather_function<<<gather_grid, block_size, 0, stream>>>(
            probs.data.as_device_ptr(),
            targets_device.as_device_ptr(),
            target_probs.data.as_device_ptr(),
            vocab_size as i32,
            batch_seq as i32
        ))?;
    }

    stream
        .synchronize()
        .context("Stream sync failed for cross-entropy kernels")?;

    let target_probs_host = target_probs.to_host()?;
    let mut loss_acc = 0f32;
    for prob in target_probs_host {
        let clipped = prob.max(1e-12);
        loss_acc += -clipped.ln();
    }

    let loss = loss_acc / batch_seq as f32;
    Ok((grad_logits, loss))
}

/// Load a UTF-8 text corpus from disk and tokenize it into a flat token stream.
///
/// The dataset is expected to be preprocessed into newline-delimited UTF-8 text.
/// Hugging Face datasets (such as `open-phi/textbooks`) can be exported to plain
/// text using the `huggingface-cli` tool or Python helpers; this function strictly
/// consumes local text files and avoids introducing additional dependencies.
pub fn load_text_corpus(path: &Path, tokenizer: &Tokenizer) -> Result<Vec<u32>> {
    let file = File::open(path).with_context(|| format!("Failed to open corpus at {:?}", path))?;
    let reader = BufReader::new(file);
    let mut tokens = Vec::new();
    for line in reader.lines() {
        let line = line?;
        if !line.is_empty() {
            tokens.extend(tokenizer.encode(&line));
        }
        // Use a newline token to delimit documents and preserve structure.
        tokens.push(b'\n' as u32);
    }

    if tokens.len() <= 1 {
        return Err(anyhow!(
            "Corpus at {:?} did not produce enough tokens for training",
            path
        ));
    }

    Ok(tokens)
}

/// Train the GPT model using a basic SGD loop with analytic gradients.
///
/// This routine performs end-to-end backpropagation through the embeddings,
/// transformer blocks, and layer norm in addition to the language-model head,
/// applying momentum SGD with optional weight decay to every parameter.
pub fn train_lm_head_from_text(
    model: &mut GPTModel,
    tokenizer: &Tokenizer,
    config: &TrainingConfig,
    corpus_path: &Path,
) -> Result<TrainingReport> {
    config.validate()?;
    let tokens = load_text_corpus(corpus_path, tokenizer)?;

    let required = config
        .seq_len
        .checked_add(1)
        .ok_or_else(|| anyhow!("seq_len {} is too large", config.seq_len))?;

    if tokens.len() < required {
        return Err(anyhow!(
            "Corpus provides {} tokens, but seq_len {} requires at least seq_len + 1",
            tokens.len(),
            config.seq_len
        ));
    }

    let window_limit = tokens.len() - required;
    let mut rng = rand::rng();

    let mut epoch_losses = Vec::with_capacity(config.epochs);
    let mut batches_per_epoch = Vec::with_capacity(config.epochs);
    let mut tokens_per_epoch = Vec::with_capacity(config.epochs);
    let mut total_batches = 0usize;
    let mut total_tokens = 0usize;

    let (embed_dim, vocab_size, has_bias) = {
        let (weight_tensor, bias_tensor_opt) = model.lm_head_params_mut();
        if weight_tensor.shape().len() != 2 {
            return Err(anyhow!(
                "LM head weight must be 2D, found shape {:?}",
                weight_tensor.shape()
            ));
        }
        let embed_dim = weight_tensor.shape()[0];
        let vocab_size = weight_tensor.shape()[1];
        let has_bias = bias_tensor_opt.is_some();
        (embed_dim, vocab_size, has_bias)
    };
    let mut lm_head_optimizer =
        LmHeadOptimizer::new(model.device(), embed_dim, vocab_size, has_bias, config)?;
    let mut encoder_optimizer = EncoderOptimizer::new(model, config)?;

    for _epoch in 0..config.epochs {
        let mut positions: Vec<usize> = (0..=window_limit).collect();
        if config.shuffle_windows {
            positions.shuffle(&mut rng);
        }
        if let Some(limit) = config.max_sequences_per_epoch {
            if positions.len() > limit {
                positions.truncate(limit);
            }
        }

        let mut epoch_loss_acc = 0f32;
        let mut epoch_batches = 0usize;
        let mut epoch_tokens = 0usize;

        for chunk in positions.chunks(config.batch_size) {
            if chunk.len() < config.batch_size {
                continue;
            }

            let mut inputs = Vec::with_capacity(config.batch_size * config.seq_len);
            let mut targets = Vec::with_capacity(config.batch_size * config.seq_len);

            for &start in chunk {
                let input_slice = &tokens[start..start + config.seq_len];
                let target_slice = &tokens[start + 1..start + config.seq_len + 1];
                inputs.extend_from_slice(input_slice);
                targets.extend_from_slice(target_slice);
            }

            let (hidden, logits, cache) =
                model.forward_with_cache(&inputs, config.batch_size, config.seq_len)?;
            let (grad_hidden, batch_loss) =
                lm_head_optimizer.step(model, hidden, logits, &targets)?;

            let encoder_grads = model.backward(&cache, &grad_hidden)?;
            encoder_optimizer.step(model, encoder_grads)?;

            epoch_loss_acc += batch_loss;
            epoch_batches += 1;
            total_batches += 1;
            epoch_tokens += config.batch_size * config.seq_len;
            total_tokens += config.batch_size * config.seq_len;

            if total_batches % config.log_every == 0 {
                println!("  batch {:>5} | loss {:.6}", total_batches, batch_loss);
            }
        }

        if epoch_batches == 0 {
            return Err(anyhow!(
                "No full batches were formed. Reduce batch_size or provide more data."
            ));
        }

        epoch_losses.push(epoch_loss_acc / epoch_batches as f32);
        batches_per_epoch.push(epoch_batches);
        tokens_per_epoch.push(epoch_tokens);
    }

    Ok(TrainingReport {
        epoch_losses,
        batches_per_epoch,
        tokens_per_epoch,
        total_batches,
        total_tokens,
    })
}
