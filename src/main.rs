use anyhow::{Context as AnyhowContext, Result, anyhow};
use cust::device::Device;
use llm_rs::dataset::download_textbooks;
use llm_rs::layers::ActivationKind;
use llm_rs::model::{GPTConfig, GPTModel};
use llm_rs::tokenizer::Tokenizer;
use llm_rs::train::{TrainingConfig, train_lm_head_from_text};
use std::path::{Path, PathBuf};

enum Command {
    Download {
        output: PathBuf,
    },
    Train {
        corpus: PathBuf,
        epochs: usize,
        save_dir: Option<PathBuf>,
    },
}

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {:#}", err);
        std::process::exit(1);
    }
}

fn run() -> Result<()> {
    let command = parse_command(&std::env::args().collect::<Vec<_>>())?;

    match command {
        Command::Download { output } => {
            download_textbooks(&output)?;
            println!("Dataset downloaded to {}", output.display());
            Ok(())
        }
        Command::Train {
            corpus,
            epochs,
            save_dir,
        } => {
            let _context = cust::quick_init().context("Failed to initialise CUDA")?;
            let device = Device::get_device(0).context("Failed to acquire CUDA device 0")?;

            run_training(&device, corpus.as_path(), epochs, save_dir.as_deref())
        }
    }
}

fn parse_command(args: &[String]) -> Result<Command> {
    let mut iter = args.iter().skip(1);
    let first = iter.next().ok_or_else(|| {
        anyhow!(
            "Usage: cargo run -- <download <output.txt> | <path/to/corpus.txt> [epochs] [--save <dir>] >"
        )
    })?;

    if first == "download" {
        let output = iter
            .next()
            .ok_or_else(|| anyhow!("Usage: cargo run -- download <output.txt>"))?;
        if iter.next().is_some() {
            return Err(anyhow!(
                "Unexpected additional arguments after download output path"
            ));
        }
        Ok(Command::Download {
            output: PathBuf::from(output),
        })
    } else {
        let corpus = PathBuf::from(first);
        let remaining: Vec<&String> = iter.collect();
        let mut idx = 0usize;

        let mut epochs = 1usize;
        let mut epochs_set = false;
        let mut save_dir = None;
        if idx < remaining.len() {
            if let Ok(parsed) = remaining[idx].parse::<usize>() {
                epochs = parsed;
                epochs_set = true;
                idx += 1;
            }
        }

        while idx < remaining.len() {
            match remaining[idx].as_str() {
                "--save" => {
                    idx += 1;
                    let path = remaining
                        .get(idx)
                        .ok_or_else(|| anyhow!("--save flag requires a directory argument"))?;
                    save_dir = Some(PathBuf::from(path));
                    idx += 1;
                }
                value if !epochs_set => {
                    epochs = value
                        .parse::<usize>()
                        .context("Could not parse epochs argument as usize")?;
                    epochs_set = true;
                    idx += 1;
                }
                unexpected => {
                    return Err(anyhow!(
                        "Unexpected argument '{}' after corpus path",
                        unexpected
                    ));
                }
            }
        }
        Ok(Command::Train {
            corpus,
            epochs,
            save_dir,
        })
    }
}

fn run_training(
    device: &Device,
    corpus_path: &Path,
    epochs: usize,
    save_dir: Option<&Path>,
) -> Result<()> {
    let tokenizer = Tokenizer::new();

    let base_learning_rate = 3e-4;

    let training_config = TrainingConfig {
        batch_size: 24,
        seq_len: 384,
        learning_rate: base_learning_rate,
        epochs,
        max_sequences_per_epoch: Some(16384),
        shuffle_windows: true,
        momentum: 0.95,
        weight_decay: 2e-2,
        log_every: 50,
        gradient_clip_norm: Some(1.0),
    };
    training_config.validate()?;

    let model_config = GPTConfig {
        vocab_size: tokenizer.vocab_size(),
        max_seq_len: training_config.seq_len,
        embed_dim: 512,
        num_heads: 8,
        num_layers: 6,
        feed_forward_dim: 2048,
        layer_norm_eps: 1e-5,
        activation: ActivationKind::Relu,
    };

    let mut model = GPTModel::new(device, model_config)?;

    let report = train_lm_head_from_text(&mut model, &tokenizer, &training_config, corpus_path)?;

    for epoch_idx in 0..report.epoch_losses.len() {
        let loss = report.epoch_losses[epoch_idx];
        let batches = report.batches_per_epoch[epoch_idx];
        let tokens = report.tokens_per_epoch[epoch_idx];
        println!(
            "Epoch {:>2} | avg loss {:.6} | batches {:>4} | tokens {:>8}",
            epoch_idx + 1,
            loss,
            batches,
            tokens
        );
    }

    println!(
        "Summary | total batches {:>4} | total tokens {:>8}",
        report.total_batches, report.total_tokens
    );

    if let Some(dir) = save_dir {
        model
            .save_checkpoint(dir)
            .with_context(|| format!("Failed to export checkpoint to {:?}", dir))?;
        println!(
            "Checkpoint saved to {:?} (config.json + model.safetensors)",
            dir
        );
    }

    Ok(())
}
