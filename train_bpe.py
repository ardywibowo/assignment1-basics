from cs336_basics.tokenizer import train_bpe

vocab, merges = train_bpe(
    input_path="data/TinyStoriesV2-GPT4-train.txt",
    vocab_size=10_000,
    special_tokens=["<|endoftext|>"],
    profile=True,
    progress=True,
)

# Optionally serialize the outputs
import json

with open("tokenizer_vocab.json", "w") as f:
    json.dump({i: token.decode("utf-8", "ignore") for i, token in vocab.items()}, f)
with open("tokenizer_merges.txt", "w") as f:
    for a, b in merges:
        f.write(f"{a.decode('utf-8', 'ignore')} {b.decode('utf-8', 'ignore')}\n")
