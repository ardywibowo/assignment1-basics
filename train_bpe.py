import argparse
import json
import os
from cs336_basics.tokenizer import train_bpe


def main():
    parser = argparse.ArgumentParser(description="Train BPE tokenizer")
    parser.add_argument(
        "--input_path", type=str, required=True, help="Path to input text file"
    )
    parser.add_argument(
        "--output_folder",
        type=str,
        required=True,
        help="Output folder for tokenizer files",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=10_000,
        help="Vocabulary size (default: 10000)",
    )

    args = parser.parse_args()

    # Create output folder if it doesn't exist
    os.makedirs(args.output_folder, exist_ok=True)

    vocab, merges = train_bpe(
        input_path=args.input_path,
        vocab_size=args.vocab_size,
        special_tokens=["<|endoftext|>"],
        profile=True,
        progress=True,
    )

    # Serialize the outputs to the specified folder
    vocab_path = os.path.join(args.output_folder, "tokenizer_vocab.json")
    merges_path = os.path.join(args.output_folder, "tokenizer_merges.txt")

    with open(vocab_path, "w") as f:
        json.dump({i: token.decode("utf-8", "ignore") for i, token in vocab.items()}, f)
    with open(merges_path, "w") as f:
        for a, b in merges:
            f.write(f"{a.decode('utf-8', 'ignore')} {b.decode('utf-8', 'ignore')}\n")


if __name__ == "__main__":
    main()
