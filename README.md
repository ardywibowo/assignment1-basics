# CS336 Spring 2025 Assignment 1: Basics

For a full description of the assignment, see the assignment handout at
[cs336_spring2025_assignment1_basics.pdf](./cs336_spring2025_assignment1_basics.pdf)

If you see any issues with the assignment handout or code, please feel free to
raise a GitHub issue or open a pull request with a fix.

## Setup

### Environment

We manage our environments with `uv` to ensure reproducibility, portability, and ease of use.
Install `uv` [here](https://github.com/astral-sh/uv) (recommended), or run `pip install uv`/`brew install uv`.
We recommend reading a bit about managing projects in `uv` [here](https://docs.astral.sh/uv/guides/projects/#managing-dependencies) (you will not regret it!).

You can now run any code in the repo using

```sh
uv run <python_file_path>
```

and the environment will be automatically solved and activated when necessary.

### Run unit tests

```sh
uv run pytest
```

Initially, all tests should fail with `NotImplementedError`s.
To connect your implementation to the tests, complete the
functions in [./tests/adapters.py](./tests/adapters.py).

### Profiling BPE training

You can inspect performance of the tokenizer training routine by enabling
profiling when calling `train_bpe` (or the `run_train_bpe` adapter):

```python
from cs336_basics import tokenizer

vocab, merges = tokenizer.train_bpe(
    input_path="path/to/corpus.txt",
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
    profile=True,
)
```

This prints a `cProfile` report summarizing where time is spent during training.

### Progress indicators

Enable a progress bar for tokenizer training with the `progress` flag:

```python
vocab, merges = tokenizer.train_bpe(
    input_path="path/to/corpus.txt",
    vocab_size=500,
    special_tokens=["<|endoftext|>"],
    progress=True,
)
```

### Download data

Download the TinyStories data and a subsample of OpenWebText

```sh
# Create and enter the data directory
mkdir -p data
cd data

# Download TinyStories dataset
curl -L -o TinyStoriesV2-GPT4-train.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
curl -L -o TinyStoriesV2-GPT4-valid.txt https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt

# Download and decompress OWT train
curl -L -o owt_train.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip -f owt_train.txt.gz

# Download and decompress OWT valid
curl -L -o owt_valid.txt.gz https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip -f owt_valid.txt.gz

# Return to original directory
cd ..

```
