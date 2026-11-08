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
# The engine reads credentials ONLY from UBT_LLM_API_KEY or a provider block's
# api_key — it never sniffs vendor variables like DEEPSEEK_API_KEY directly.
# Declare the provider once, then select it (both steps live in the sample
# config, which was written for exactly this flow):
cp examples/sample_book/ubt_sample_config.toml ubt.toml
export DEEPSEEK_API_KEY="sk-..."
ubt translate examples/sample_book/deep_learning_primer.md -o output/deep_learning_zh.pdf -l zh

# Local Ollama (100% offline & free): uncomment the [providers.ollama] block in
# the same file, then select it explicitly. Vendor names are not built-in
# presets — an undeclared one fails with ProviderNotFoundError.
ubt translate examples/sample_book/deep_learning_primer.md -o output/deep_learning_zh.pdf -l zh --provider ollama --model qwen2.5:14b
```

Configuration precedence is `CLI flag > env (UBT_*) > [providers.<name>] > [defaults] > default`; the credential variables the engine recognises are listed in [`ubt.toml.example`](../../ubt.toml.example).
