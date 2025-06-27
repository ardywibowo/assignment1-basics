# CS336 Assignment 1 Setup Guide

This guide helps you set up the development environment for CS336 Assignment 1, including the high-performance Rust BPE tokenizer.

## Quick Setup (Recommended)

Run the automated setup script:

```bash
# Clone and enter the repository
git clone <repository-url>
cd assignment1-basics

# Run the setup script
./setup.sh
```

The script will automatically:
- Install Rust (if not already installed)
- Install uv Python package manager (if not already installed)
- Set up Python virtual environment and dependencies
- Build and install the Rust BPE tokenizer
- Download required datasets
- Run tests to verify installation

## Manual Setup

If you prefer to set up manually or the automatic script fails:

### 1. Install Prerequisites

**Rust** (required for fast BPE tokenizer):
```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

**Python 3.11+** (required):
- macOS: `brew install python@3.11`
- Ubuntu: `sudo apt install python3.11 python3.11-venv`
- Or use pyenv, conda, etc.

**uv** (Python package manager):
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 2. Set Up Python Environment

```bash
# Install Python dependencies
uv sync
```

### 3. Build Rust BPE Tokenizer

```bash
# Build and install the Rust tokenizer
cd bpe
uv run maturin develop --release
cd ..
```

### 4. Verify Installation

```bash
# Test the installation
uv run python -c "
import tokenizer_rs
from cs336_basics.tokenizer import train_bpe
print('✅ Installation successful!')
"

# Run full test suite
uv run python -m pytest tests/
```

### 5. Download Datasets (Optional)

```bash
mkdir -p data
cd data

# TinyStories dataset
curl -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# OpenWebText dataset
curl -L -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
curl -L -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_train.txt.gz owt_valid.txt.gz

cd ..
```

## Usage Examples

### BPE Tokenizer Training

**Fast Rust version with progress bars:**
```python
from cs336_basics.tokenizer import train_bpe

vocab, merges = train_bpe(
    input_path='data/TinyStoriesV2-GPT4-train.txt',
    vocab_size=1000,
    special_tokens=['<|endoftext|>'],
    progress=True,    # Show progress bars
    use_rust=True     # Use fast Rust implementation
)

print(f"Trained tokenizer with {len(vocab)} vocabulary items")
```

**Python version (slower but compatible):**
```python
vocab, merges = train_bpe(
    input_path='data/TinyStoriesV2-GPT4-train.txt',
    vocab_size=1000,
    special_tokens=['<|endoftext|>'],
    progress=True,
    use_rust=False    # Use Python implementation
)
```

### Development Commands

```bash
# Run specific tests
uv run python -m pytest tests/test_train_bpe.py -v

# Run with coverage
uv run python -m pytest tests/ --cov=cs336_basics

# Start Python REPL with dependencies
uv run python

# Run linting
uv run ruff check cs336_basics/
```

## Troubleshooting

### Rust Compilation Issues

If the Rust build fails, try:
```bash
# Update Rust
rustup update

# Clean and rebuild
cd bpe
cargo clean
uv run maturin develop --release
cd ..
```

### Import Errors

If you can't import `tokenizer_rs`:
```bash
# Rebuild the tokenizer from the bpe directory
cd bpe
uv run maturin develop --release
cd ..
```

### Performance Comparison

The Rust BPE tokenizer provides significant speedup:
- **Rust**: ~10-50x faster than Python implementation
- **Python**: More compatible, easier to debug

Use `use_rust=True` for production training, `use_rust=False` for debugging.

## Features

- ✅ **Fast Rust BPE implementation** with progress bars
- ✅ **Parameterized training** (vocab_size, special_tokens)
- ✅ **UTF-8 safe** handling of all byte sequences
- ✅ **Compatible interface** between Python and Rust versions
- ✅ **Comprehensive test suite** ensuring correctness
- ✅ **Progress indicators** for long-running operations

## Architecture

```
assignment1-basics/
├── cs336_basics/          # Main Python package
│   └── tokenizer.py       # BPE interface and Python implementation
├── bpe/                   # Rust BPE tokenizer
│   ├── src/
│   │   ├── lib.rs         # Python interface
│   │   ├── bpe.rs         # Core BPE algorithm
│   │   └── pretokenize.rs # Text preprocessing
│   └── Cargo.toml         # Rust dependencies
├── tests/                 # Test suite
└── setup.sh              # Automated setup script
```

The Rust tokenizer is built using [PyO3](https://pyo3.rs/) for Python bindings and [maturin](https://www.maturin.rs/) for building.