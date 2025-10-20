# llm-rs

A simple GPT-based LLM implemented mainly in Rust. CUDA work happens through `cust`, `cublas`, and `cuda_std`, while bespoke CUDA C++ kernels power the heavy lifting.


Mostly, this was done for learning purposes to explore LLMs. The resulting model is usable, but it wasnt benchmarked and isnt anything special. You can check out the model weights [here](https://huggingface.co/zekdevs/llm-rs)

## Dataset

The project targets the [open-phi/textbooks](https://huggingface.co/datasets/open-phi/textbooks) corpus.

### Downloading Datasets

The repository now ships with a lightweight downloader that speaks directly to
the Hugging Face datasets-server API and emits newline-delimited UTF-8:

```bash
cargo run --release -- download ./textbooks.txt
```

## Training

The binary performs momentum SGD across the entire transformer stack—token and position embeddings, every transformer block, the final layer norm, and the language-model head—using analytic CUDA-backed gradients. Provide the corpus path and optionally the number of epochs. Pass `--save <dir>` to export the trained weights:

```bash
cargo run --release -- ./textbooks.txt 3 --save ./checkpoints/baseline
```

### Saving weights

When `--save` is set the binary ensures the directory exists and writes two files compatible with Hugging Face uploads:

- `config.json` – serialized `GPTConfig`
- `model.safetensors` – parameter tensors in the [safetensors](https://github.com/huggingface/safetensors) format
