# Quick Start

## One-Line Setup

```bash
curl -sSL https://raw.githubusercontent.com/your-repo/assignment1-basics/main/setup.sh | bash
```

Or clone first:

```bash
git clone <repository-url>
cd assignment1-basics
./setup.sh
```

## What Gets Installed

- 🦀 **Rust** (if not present) - for the fast BPE tokenizer
- 📦 **uv** (if not present) - modern Python package manager  
- 🐍 **Python dependencies** - PyTorch, numpy, regex, etc.
- ⚡ **Rust BPE tokenizer** - 10-50x faster than Python implementation
- 📊 **Datasets** - TinyStories and OpenWebText samples
- 🧪 **Test verification** - ensures everything works

## Immediate Usage

After setup, train a BPE tokenizer:

```python
from cs336_basics.tokenizer import train_bpe

# Fast Rust version with progress bars
vocab, merges = train_bpe(
    'data/TinyStoriesV2-GPT4-train.txt',
    vocab_size=1000,
    special_tokens=['<|endoftext|>'],
    progress=True,    # Show progress bars  
    use_rust=True     # 10-50x faster!
)
```

## Requirements

- **macOS/Linux** (Windows via WSL)
- **curl** (for downloading)
- **Python 3.11+** (will warn if older)
- **Internet connection** (for downloads)

The script handles everything else automatically!