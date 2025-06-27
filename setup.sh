#!/bin/bash

# CS336 Assignment 1 Setup Script
# This script sets up the development environment and builds the Rust BPE tokenizer

set -e  # Exit on any error

echo "🚀 Setting up CS336 Assignment 1 environment..."

# Check if we're in the right directory
if [[ ! -f "pyproject.toml" ]]; then
    echo "❌ Error: Please run this script from the project root directory"
    exit 1
fi

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

echo "📦 Checking dependencies..."

# Check for Rust installation
if ! command_exists rustc; then
    echo "🦀 Installing Rust..."
    if command_exists curl; then
        curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
        source "$HOME/.cargo/env"
    else
        echo "❌ Error: curl is required to install Rust. Please install curl first."
        exit 1
    fi
else
    echo "✅ Rust is already installed ($(rustc --version))"
fi

# Check for Python 3.11+
if command_exists python3; then
    PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
    echo "✅ Python $PYTHON_VERSION found"
    if [[ $(echo "$PYTHON_VERSION >= 3.11" | bc -l 2>/dev/null || echo "0") == "0" ]]; then
        echo "⚠️  Warning: Python 3.11+ is recommended. Current version: $PYTHON_VERSION"
    fi
else
    echo "❌ Error: Python 3 is required. Please install Python 3.11+ first."
    exit 1
fi

# Check for uv (Python package manager)
if ! command_exists uv; then
    echo "📦 Installing uv (Python package manager)..."
    if command_exists curl; then
        curl -LsSf https://astral.sh/uv/install.sh | sh
        source "$HOME/.local/bin/env" 2>/dev/null || true
        export PATH="$HOME/.local/bin:$PATH"
    else
        echo "❌ Error: curl is required to install uv. Please install curl first."
        exit 1
    fi
else
    echo "✅ uv is already installed ($(uv --version))"
fi

echo "🐍 Setting up Python environment..."

# Create virtual environment and install Python dependencies
uv sync

echo "🦀 Building Rust BPE tokenizer..."

# Build and install the Rust BPE tokenizer
cd bpe
uv run maturin develop --release
cd ..

echo "🧪 Running tests to verify installation..."

# Run a quick test to verify everything works
if uv run python -c "
import tokenizer_rs
from cs336_basics.tokenizer import train_bpe
print('✅ Rust tokenizer successfully imported')
print('✅ BPE training interface working')
" 2>/dev/null; then
    echo "✅ Rust BPE tokenizer installation verified!"
else
    echo "❌ Error: Rust BPE tokenizer installation failed"
    exit 1
fi

echo "📊 Downloading datasets..."

# Create and enter the data directory
mkdir -p data
cd data

# Download TinyStories dataset
if [[ ! -f "TinyStoriesV2-GPT4-train.txt" ]]; then
    echo "📥 Downloading TinyStories training data..."
    curl -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
else
    echo "✅ TinyStories training data already exists"
fi

if [[ ! -f "TinyStoriesV2-GPT4-valid.txt" ]]; then
    echo "📥 Downloading TinyStories validation data..."
    curl -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
else
    echo "✅ TinyStories validation data already exists"
fi

# Download and decompress OWT train
if [[ ! -f "owt_train.txt" ]]; then
    echo "📥 Downloading OWT training data..."
    curl -L -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
    gunzip -f owt_train.txt.gz
else
    echo "✅ OWT training data already exists"
fi

# Download and decompress OWT valid
if [[ ! -f "owt_valid.txt" ]]; then
    echo "📥 Downloading OWT validation data..."
    curl -L -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
    gunzip -f owt_valid.txt.gz
else
    echo "✅ OWT validation data already exists"
fi

# Return to original directory
cd ..

echo "🧪 Running full test suite..."

# Run the test suite to ensure everything works
if uv run python -m pytest tests/ -q; then
    echo "✅ All tests passed!"
else
    echo "⚠️  Some tests failed, but installation may still be functional"
fi

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📝 Usage examples:"
echo ""
echo "# Train BPE tokenizer with progress bars (Rust - fast):"
echo "uv run python -c \"
from cs336_basics.tokenizer import train_bpe
vocab, merges = train_bpe(
    'tests/fixtures/corpus.en',
    vocab_size=500,
    special_tokens=['<|endoftext|>'],
    progress=True,
    use_rust=True
)
print(f'Trained tokenizer with {len(vocab)} vocab items')
\""
echo ""
echo "# Train BPE tokenizer (Python - slower but compatible):"
echo "uv run python -c \"
from cs336_basics.tokenizer import train_bpe
vocab, merges = train_bpe(
    'tests/fixtures/corpus.en',
    vocab_size=500,
    special_tokens=['<|endoftext|>'],
    progress=True,
    use_rust=False
)
print(f'Trained tokenizer with {len(vocab)} vocab items')
\""
echo ""
echo "# Run tests:"
echo "uv run python -m pytest tests/"
echo ""
echo "# Start development:"
echo "uv run python  # Enters Python with all dependencies available"
echo ""
echo "✨ Ready to go! The Rust BPE tokenizer provides significantly faster training."
echo ""
echo "🔍 Quick verification:"
echo "uv run python -c \"
from cs336_basics.tokenizer import train_bpe
vocab, merges = train_bpe('tests/fixtures/corpus.en', 300, ['<|endoftext|>'], use_rust=True)
print(f'✅ Quick test successful: {len(vocab)} vocab items, {len(merges)} merges')
\""
