#!/usr/bin/env python3
"""
Script to load a trained BPE tokenizer and encode a text file, saving as NumPy uint16 array.
"""
import argparse
import json
import numpy as np
from cs336_basics.tokenizer import Tokenizer


def load_tokenizer(tokenizer_folder: str, special_tokens: list[str] = None) -> Tokenizer:
    """Load a BPE tokenizer from vocab and merges files, reconstructing missing merged tokens."""
    import os
    
    vocab_path = os.path.join(tokenizer_folder, 'tokenizer_vocab.json')
    merges_path = os.path.join(tokenizer_folder, 'tokenizer_merges.txt')
    
    # Load vocabulary
    with open(vocab_path, 'r', encoding='utf-8') as f:
        vocab_json = json.load(f)
    
    # Convert string keys to int and string values to bytes
    # Fix issue where bytes 128-255 map to empty strings
    vocab = {}
    for k, v in vocab_json.items():
        key = int(k)
        if key < 256:
            # For base bytes, ensure they map to the actual byte value
            vocab[key] = bytes([key])
        else:
            # For merged tokens, use the stored value
            vocab[key] = v.encode('utf-8')
    
    # Load merges and reconstruct missing tokens
    merges = []
    next_id = max(vocab.keys()) + 1
    
    with open(merges_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                # Split only on first space to handle tokens with spaces
                parts = line.split(' ', 1)
                if len(parts) == 2:
                    a, b = parts
                    a_bytes = a.encode('utf-8')
                    b_bytes = b.encode('utf-8')
                    merged_bytes = a_bytes + b_bytes
                    
                    # Check if merged token exists in vocab, if not add it
                    if merged_bytes not in vocab.values():
                        vocab[next_id] = merged_bytes
                        next_id += 1
                    
                    merges.append((a_bytes, b_bytes))
    
    print(f"Loaded tokenizer with {len(vocab)} tokens and {len(merges)} merges")
    return Tokenizer(vocab, merges, special_tokens or [])


def main():
    parser = argparse.ArgumentParser(description="Encode text file using trained BPE tokenizer")
    parser.add_argument("--tokenizer_folder", type=str, required=True,
                       help="Path to folder containing tokenizer_vocab.json and tokenizer_merges.txt")
    parser.add_argument("--input_path", type=str, required=True,
                       help="Path to text file to encode")
    parser.add_argument("--output_path", type=str, required=True,
                       help="Path to save encoded NumPy array (.npy)")
    parser.add_argument("--special_tokens", nargs='*', default=["<|endoftext|>"],
                       help="Special tokens used during training")
    
    args = parser.parse_args()
    
    # Load the tokenizer
    print(f"Loading tokenizer from {args.tokenizer_folder}")
    tokenizer = load_tokenizer(args.tokenizer_folder, args.special_tokens)
    
    # Read the input text file
    print(f"Reading input file: {args.input_path}")
    with open(args.input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Encode the text
    print("Encoding text...")
    token_ids = tokenizer.encode(text)
    
    # Convert to NumPy array with uint16 datatype
    print(f"Converting {len(token_ids)} tokens to uint16 array")
    
    # Check if any token ID exceeds uint16 range
    max_id = max(token_ids) if token_ids else 0
    if max_id > 65535:
        print(f"Warning: Maximum token ID {max_id} exceeds uint16 range (0-65535)")
        print("Consider using uint32 datatype instead")
        # Clip values to uint16 range
        token_ids = [min(id, 65535) for id in token_ids]
    
    encoded_array = np.array(token_ids, dtype=np.uint16)
    
    # Save the array
    print(f"Saving encoded array to: {args.output_path}")
    np.save(args.output_path, encoded_array)
    
    print(f"Successfully encoded {len(token_ids)} tokens")
    print(f"Array shape: {encoded_array.shape}")
    print(f"Array dtype: {encoded_array.dtype}")
    print(f"File saved: {args.output_path}")


if __name__ == "__main__":
    main()