use crate::model::GPTModel;
use crate::tensor::Tensor;
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result, anyhow};
use rand::seq::SliceRandom;
use std::convert::TryFrom;
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
            let batch_loss = sgd_step(model, hidden, logits, &targets, config.learning_rate)?;

            epoch_loss_acc += batch_loss;
            epoch_batches += 1;
            total_batches += 1;
            epoch_tokens += config.batch_size * config.seq_len;
            total_tokens += config.batch_size * config.seq_len;
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

fn sgd_step(
    model: &mut GPTModel,
    hidden: Tensor,
    logits: Tensor,
    targets: &[u32],
    learning_rate: f32,
) -> Result<f32> {
    if hidden.shape().len() != 2 || logits.shape().len() != 2 {
        return Err(anyhow!("Expected 2D tensors for hidden states and logits"));
    }

    let batch_seq = hidden.shape()[0];
    let embed_dim = hidden.shape()[1];
    let vocab_size = logits.shape()[1];

    if logits.shape()[0] != batch_seq {
        return Err(anyhow!(
            "Hidden state count ({}) must match logits rows ({})",
            batch_seq,
            logits.shape()[0]
        ));
    }

    if targets.len() != batch_seq {
        return Err(anyhow!(
            "Target length {} does not match batch*seq {}",
            targets.len(),
            batch_seq
        ));
    }

    let hidden_host = hidden.to_host()?;
    let logits_host = logits.to_host()?;
    let mut grad_logits = vec![0f32; logits_host.len()];
    let mut loss_acc = 0f32;
    let inv_batch = 1.0f32 / (batch_seq as f32);

    for i in 0..batch_seq {
        let offset = i * vocab_size;
        let slice = &logits_host[offset..offset + vocab_size];
        let target_idx = usize::try_from(targets[i])
            .map_err(|_| anyhow!("Target token index {} does not fit in usize", targets[i]))?;
        if target_idx >= vocab_size {
            return Err(anyhow!(
                "Target token {} exceeds vocabulary size {}",
                target_idx,
                vocab_size
            ));
        }

        let max_logit = slice
            .iter()
            .fold(f32::NEG_INFINITY, |acc, &val| acc.max(val));
        let mut sum_exp = 0f32;
        for &logit in slice {
            sum_exp += (logit - max_logit).exp();
        }
        let log_sum = sum_exp.ln();
        loss_acc += -(slice[target_idx] - max_logit) + log_sum;

        for j in 0..vocab_size {
            let prob = (slice[j] - max_logit).exp() / sum_exp;
            let indicator = if j == target_idx { 1.0 } else { 0.0 };
            grad_logits[offset + j] = (prob - indicator) * inv_batch;
        }
    }

    loss_acc /= batch_seq as f32;

    let mut grad_weight = vec![0f32; embed_dim * vocab_size];
    let mut grad_bias = vec![0f32; vocab_size];

    for i in 0..batch_seq {
        let hidden_row = &hidden_host[i * embed_dim..(i + 1) * embed_dim];
        let grad_row = &grad_logits[i * vocab_size..(i + 1) * vocab_size];

        for j in 0..vocab_size {
            let grad = grad_row[j];
            grad_bias[j] += grad;
            if grad == 0.0 {
                continue;
            }

            let mut weight_offset = j;
            for &hidden_val in hidden_row {
                grad_weight[weight_offset] += hidden_val * grad;
                weight_offset += vocab_size;
            }
        }
    }

    let (weight_tensor, bias_tensor_opt) = model.lm_head_params_mut();
    let mut weight_host = weight_tensor.to_host()?;
    for (w, grad) in weight_host.iter_mut().zip(grad_weight.iter()) {
        *w -= learning_rate * grad;
    }
    weight_tensor.copy_from_host(&weight_host)?;

    if let Some(bias_tensor) = bias_tensor_opt {
        let mut bias_host = bias_tensor.to_host()?;
        for (b, grad) in bias_host.iter_mut().zip(grad_bias.iter()) {
            *b -= learning_rate * grad;
        }
        bias_tensor.copy_from_host(&bias_host)?;
    }

    Ok(loss_acc)
}
