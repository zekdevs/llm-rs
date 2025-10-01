use crate::kernel_cache;
use crate::model::GPTModel;
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow};
use cust::prelude::*;
use cust::{
    device::Device,
    memory::DeviceBuffer,
    stream::{Stream, StreamFlags},
};
use rand::{Rng, seq::SliceRandom};
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
    pub use_spsa: bool,
    pub spsa_learning_rate: f32,
    pub spsa_epsilon: f32,
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
        if self.use_spsa {
            if !(self.spsa_learning_rate.is_finite()) || self.spsa_learning_rate <= 0.0 {
                return Err(anyhow!(
                    "TrainingConfig spsa_learning_rate must be finite and positive"
                ));
            }
            if !(self.spsa_epsilon.is_finite()) || self.spsa_epsilon <= 0.0 {
                return Err(anyhow!(
                    "TrainingConfig spsa_epsilon must be finite and positive"
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
            use_spsa: false,
            spsa_learning_rate: 1e-3,
            spsa_epsilon: 1e-3,
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
        })
    }

    fn step(
        &mut self,
        model: &mut GPTModel,
        hidden: Tensor,
        logits: Tensor,
        targets: &[u32],
    ) -> Result<f32> {
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

        let hidden_t = hidden.transpose2d()?;
        let mut grad_weight = hidden_t.matmul(&grad_logits)?;

        let (weight_tensor, bias_tensor_opt) = model.lm_head_params_mut();
        let weight_decay_term = weight_tensor.mul_scalar(self.weight_decay)?;
        grad_weight = grad_weight.add(&weight_decay_term)?;

        let momentum_velocity = self.velocity_weight.mul_scalar(self.momentum)?;
        self.velocity_weight = momentum_velocity.add(&grad_weight)?;
        let weight_update = self.velocity_weight.mul_scalar(self.learning_rate)?;
        let new_weight = weight_tensor.sub(&weight_update)?;
        *weight_tensor = new_weight;

        if let Some(bias_tensor) = bias_tensor_opt {
            if let Some(velocity_bias) = &mut self.velocity_bias {
                let grad_logits_t = grad_logits.transpose2d()?;
                let grad_bias = grad_logits_t.sum_rows()?.reshape(vec![1, vocab_size])?;

                let momentum_bias = velocity_bias.mul_scalar(self.momentum)?;
                *velocity_bias = momentum_bias.add(&grad_bias)?;
                let bias_update = velocity_bias.mul_scalar(self.learning_rate)?;
                let new_bias = bias_tensor.sub(&bias_update)?;
                *bias_tensor = new_bias;
            }
        }

        Ok(loss)
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

fn cross_entropy_loss_from_probs(probs: &Tensor, targets: &[u32]) -> Result<f32> {
    if probs.shape().len() != 2 {
        return Err(anyhow!(
            "cross_entropy_loss_from_probs expects a 2D tensor, got {:?}",
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
        .context("Failed to copy targets to device for cross-entropy loss")?;

    let module = kernel_cache::module(include_str!("cross_entropy_kernel.ptx"), "cross_entropy")?;
    let gather_function = module
        .get_function("gather_target_prob_kernel")
        .context("Kernel load failed for gather_target_prob_kernel")?;

    let target_probs = Tensor::new(vec![batch_seq], &probs.device)?;
    let block_size = 256u32;
    let grid_size = ((batch_seq as u32) + block_size - 1) / block_size;
    let stream =
        Stream::new(StreamFlags::NON_BLOCKING, None).context("Failed to create CUDA stream")?;

    unsafe {
        launch!(gather_function<<<grid_size, block_size, 0, stream>>>(
            probs.data.as_device_ptr(),
            targets_device.as_device_ptr(),
            target_probs.data.as_device_ptr(),
            vocab_size as i32,
            batch_seq as i32
        ))?;
    }

    stream
        .synchronize()
        .context("Stream sync failed for gather_target_prob_kernel")?;

    let target_probs_host = target_probs.to_host()?;
    let mut loss_acc = 0f32;
    for prob in target_probs_host {
        let clipped = prob.max(1e-12);
        loss_acc += -clipped.ln();
    }

    Ok(loss_acc / batch_seq as f32)
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

/// Train only the language-model head of the GPT model using a basic SGD loop.
///
/// The remainder of the transformer stack remains frozen—this keeps the example
/// lightweight while still demonstrating end-to-end data movement, loss
/// computation, and parameter updates on the GPU-backed tensors.
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
    let mut optimizer =
        LmHeadOptimizer::new(model.device(), embed_dim, vocab_size, has_bias, config)?;

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

            let (hidden, logits) =
                model.forward_with_hidden(&inputs, config.batch_size, config.seq_len)?;
            let mut batch_loss = optimizer.step(model, hidden, logits, &targets)?;

            if config.use_spsa {
                batch_loss = spsa_step(
                    model,
                    &inputs,
                    &targets,
                    config.batch_size,
                    config.seq_len,
                    config,
                )?;
            }

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

struct ParameterSnapshot {
    name: String,
    tensor_ptr: *mut Tensor,
    original: Vec<f32>,
    delta: Vec<f32>,
}

impl ParameterSnapshot {
    fn new(name: String, tensor: &mut Tensor) -> Result<Self> {
        let original = tensor
            .to_host()
            .with_context(|| format!("Failed to copy parameter {name} to host for SPSA"))?;
        Ok(Self {
            name,
            tensor_ptr: tensor as *mut Tensor,
            original,
            delta: Vec::new(),
        })
    }

    fn with_scaled(&self, scale: f32) -> Vec<f32> {
        self.original
            .iter()
            .zip(&self.delta)
            .map(|(&weight, &direction)| weight + scale * direction)
            .collect()
    }

    fn updated_values(&self, coeff: f32, lr: f32, weight_decay: f32) -> Vec<f32> {
        self.original
            .iter()
            .zip(&self.delta)
            .map(|(&weight, &direction)| {
                let grad_estimate = coeff * direction + weight_decay * weight;
                weight - lr * grad_estimate
            })
            .collect()
    }

    fn apply(&self, data: &[f32]) -> Result<()> {
        unsafe {
            (*self.tensor_ptr)
                .copy_from_host(data)
                .with_context(|| format!("Failed to upload parameter {} during SPSA", self.name))
        }
    }
}

fn spsa_step(
    model: &mut GPTModel,
    inputs: &[u32],
    targets: &[u32],
    batch_size: usize,
    seq_len: usize,
    config: &TrainingConfig,
) -> Result<f32> {
    let mut snapshots: Vec<ParameterSnapshot> = Vec::new();
    model.visit_parameters_mut(|name, tensor| {
        if name.starts_with("lm_head") {
            return Ok(());
        }
        let snapshot = ParameterSnapshot::new(name.to_string(), tensor)?;
        snapshots.push(snapshot);
        Ok(())
    })?;

    if snapshots.is_empty() {
        return compute_batch_loss(model, inputs, targets, batch_size, seq_len);
    }

    let mut rng = rand::rng();

    for snapshot in &mut snapshots {
        snapshot.delta = snapshot
            .original
            .iter()
            .map(|_| if rng.random::<bool>() { 1.0 } else { -1.0 })
            .collect();
    }

    for snapshot in &snapshots {
        let plus = snapshot.with_scaled(config.spsa_epsilon);
        snapshot.apply(&plus)?;
    }
    let loss_plus = compute_batch_loss(model, inputs, targets, batch_size, seq_len)?;

    for snapshot in &snapshots {
        let minus = snapshot.with_scaled(-config.spsa_epsilon);
        snapshot.apply(&minus)?;
    }
    let loss_minus = compute_batch_loss(model, inputs, targets, batch_size, seq_len)?;

    for snapshot in &snapshots {
        snapshot.apply(&snapshot.original)?;
    }

    let coeff = (loss_plus - loss_minus) / (2.0 * config.spsa_epsilon);

    for snapshot in &snapshots {
        let updated =
            snapshot.updated_values(coeff, config.spsa_learning_rate, config.weight_decay);
        snapshot.apply(&updated)?;
    }

    compute_batch_loss(model, inputs, targets, batch_size, seq_len)
}

fn compute_batch_loss(
    model: &GPTModel,
    inputs: &[u32],
    targets: &[u32],
    batch_size: usize,
    seq_len: usize,
) -> Result<f32> {
    let vocab_size = model.config().vocab_size;
    let logits = model.forward(inputs, batch_size, seq_len)?;
    let logits = logits.reshape(vec![batch_size * seq_len, vocab_size])?;
    let probs = logits.softmax()?;
    cross_entropy_loss_from_probs(&probs, targets)
}
