# llm-rs

A simple GPT-based LLM implemented mainly in Rust. CUDA work happens through `cust`, `cublas`, and `cuda_std`, while bespoke CUDA C++ kernels power the heavy lifting.

C++ is used for the CUDA kernels.
Python is used to convert the weights to PyTorch state_dicts.

All OSS. Trained on FAL hardware.

## Dataset

The project targets the [open-phi/textbooks](https://huggingface.co/datasets/open-phi/textbooks) corpus.

### Exporting to plain text

The repository now ships with a lightweight downloader that speaks directly to
the Hugging Face datasets-server API and emits newline-delimited UTF-8:

```bash
cargo run --release -- download ./textbooks.txt
```

Each request fetches a batch of rows, strips embedded newlines, and appends a
single paragraph per line. Point the training binary at the resulting
`textbooks.txt` file to kick off training.

## Training entry point

The default binary now performs a lightweight SGD step that updates only the language-model head while the rest of the transformer remains frozen. Provide the corpus path and optionally the number of epochs:

```bash
cargo run --release -- ./textbooks.txt 3
```

The training loop emits per-epoch average loss alongside aggregate token and batch counts. It uses only the dependencies listed in `Cargo.toml`.

## Notes and next steps

- Extend the SGD step with additional CUDA kernels before fine-tuning deeper layers.
- Add proper batching, shuffling, and checkpointing for large-scale runs.
- Replace the placeholder dataset export with a more scalable preprocessing pipeline before training on the full corpus.
