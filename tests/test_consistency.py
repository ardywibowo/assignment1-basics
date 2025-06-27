#!/usr/bin/env python3

import os
import sys
sys.path.append('/Users/randyardywibowo/Github/assignment1-basics')

from cs336_basics.tokenizer import train_bpe

def test_consistency():
    # Create a small test file with multiple documents
    test_content = """This is document 1. It has some text.<|endoftext|>This is document 2 with different content.<|endoftext|>Document 3 is here with more text and symbols like !@# and numbers 123.<|endoftext|>Final document 4 has unicode: café résumé naïve."""
    
    test_file = "/tmp/test_consistency.txt"
    with open(test_file, 'w', encoding='utf-8') as f:
        f.write(test_content)
    
    vocab_size = 300
    special_tokens = ["<|endoftext|>"]
    
    print("Testing regular vs document-based chunking consistency...")
    
    # Test 1: Regular processing (small file, no chunking)
    print("1. Running regular processing...")
    vocab1, merges1 = train_bpe(
        test_file, 
        vocab_size, 
        special_tokens, 
        use_rust=True, 
        force_chunked=False,
        progress=True
    )
    
    # Test 2: Forced chunked processing 
    print("2. Running document-based chunked processing...")
    vocab2, merges2 = train_bpe(
        test_file, 
        vocab_size, 
        special_tokens, 
        use_rust=True, 
        force_chunked=True,
        progress=True
    )
    
    # Compare results
    print("3. Comparing results...")
    
    # Check if vocabularies are identical
    if vocab1 == vocab2:
        print("✓ Vocabularies are identical")
    else:
        print("✗ Vocabularies differ")
        print(f"Regular vocab size: {len(vocab1)}, Chunked vocab size: {len(vocab2)}")
        # Find differences
        keys1, keys2 = set(vocab1.keys()), set(vocab2.keys())
        if keys1 != keys2:
            print(f"Key differences: {keys1.symmetric_difference(keys2)}")
        for k in keys1.intersection(keys2):
            if vocab1[k] != vocab2[k]:
                print(f"Value diff at key {k}: {vocab1[k]} vs {vocab2[k]}")
    
    # Check if merges are identical  
    if merges1 == merges2:
        print("✓ Merge sequences are identical")
    else:
        print("✗ Merge sequences differ")
        print(f"Regular merges: {len(merges1)}, Chunked merges: {len(merges2)}")
        if len(merges1) == len(merges2):
            for i, (m1, m2) in enumerate(zip(merges1, merges2)):
                if m1 != m2:
                    print(f"First difference at merge {i}: {m1} vs {m2}")
                    break
    
    # Clean up
    os.remove(test_file)
    
    # Assert that both processing methods produce identical results
    assert vocab1 == vocab2, "Vocabularies differ between regular and chunked processing"
    assert merges1 == merges2, "Merge sequences differ between regular and chunked processing"
    
    print("\n🎉 SUCCESS: Both processing methods produce identical results!")

if __name__ == "__main__":
    try:
        test_consistency()
        sys.exit(0)
    except AssertionError as e:
        print(f"\n❌ FAILURE: {e}")
        sys.exit(1)