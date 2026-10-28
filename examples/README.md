# UBT Examples & Reference Artifacts

This directory contains reference sample documents, configuration templates, and verification scripts for the Universal Book Translator (UBT).

## Contents

- `sample_book/deep_learning_primer.md`: A mathematically rigorous sample mini-monograph covering neural representations, loss functions, complexity analysis tables, and PyTorch execution kernels.
- `sample_book/ubt_sample_config.toml`: An annotated reference configuration demonstrating provider selection, delivery mode options, layout auto-healing settings, and domain glossary rules.

## Quick Start: Testing the Reference Book

Run the self-contained verification runner (offline, deterministic, zero-token cost):

```bash
uv run python scripts/verify_sample_book.py
```

To translate the sample book with a live provider (e.g. DeepSeek or local Ollama):

```bash
# DeepSeek API
export DEEPSEEK_API_KEY="sk-..."
ubt translate examples/sample_book/deep_learning_primer.md -o output/deep_learning_zh.pdf -l zh

# Local Ollama (100% offline & free)
ubt translate examples/sample_book/deep_learning_primer.md -o output/deep_learning_zh.pdf -l zh --provider ollama --model qwen2.5:14b
```
