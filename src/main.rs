use anyhow::{Context as AnyhowContext, Result, anyhow};
use cust::device::Device;
use llm_rs::dataset::download_textbooks;
use llm_rs::layers::ActivationKind;
use llm_rs::model::{GPTConfig, GPTModel};
use llm_rs::tokenizer::Tokenizer;
use llm_rs::train::{TrainingConfig, train_lm_head_from_text};
use std::path::{Path, PathBuf};

enum Command {
    Download { output: PathBuf },
    Train { corpus: PathBuf, epochs: usize },
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
        Command::Train { corpus, epochs } => {
            let _context = cust::quick_init().context("Failed to initialise CUDA")?;
            let device = Device::get_device(0).context("Failed to acquire CUDA device 0")?;

            run_training(&device, corpus.as_path(), epochs)
        }
    }
}

fn parse_command(args: &[String]) -> Result<Command> {
    let mut iter = args.iter().skip(1);
    let first = iter.next().ok_or_else(|| {
        anyhow!("Usage: cargo run -- <download <output.txt> | <path/to/corpus.txt> [epochs]>")
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
        let epochs = match iter.next() {
            Some(value) => value
                .parse::<usize>()
                .context("Could not parse epochs argument as usize")?,
            None => 1,
        };
        if iter.next().is_some() {
            return Err(anyhow!(
                "Unexpected additional arguments after epochs value"
            ));
        }
        Ok(Command::Train { corpus, epochs })
    }
}

fn run_training(device: &Device, corpus_path: &Path, epochs: usize) -> Result<()> {
    let tokenizer = Tokenizer::new();

    let training_config = TrainingConfig {
        batch_size: 4,
        seq_len: 128,
        learning_rate: 1e-3,
        epochs,
        max_sequences_per_epoch: Some(2048),
        shuffle_windows: true,
    };
    training_config.validate()?;

    let model_config = GPTConfig {
        vocab_size: tokenizer.vocab_size(),
        max_seq_len: training_config.seq_len,
        embed_dim: 128,
        num_heads: 4,
        num_layers: 2,
        feed_forward_dim: 512,
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

    Ok(())
}
