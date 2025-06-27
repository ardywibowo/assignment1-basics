from __future__ import annotations

import cProfile
import multiprocessing as mp
import pstats
from collections import Counter
from io import StringIO
from collections.abc import Iterable

import regex as re
from tqdm import tqdm, trange

try:
    from tokenizer_rs import train_bpe_py as _rust_train_bpe
    from tokenizer_rs import train_bpe_from_file_py as _rust_train_bpe_from_file
except Exception:  # pragma: no cover - optional dependency may not be built
    _rust_train_bpe = None
    _rust_train_bpe_from_file = None

PATTERN = re.compile(
    r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
)


def _count_tokens(text: str) -> Counter[tuple[bytes, ...]]:
    """Count byte-level tokens in a chunk of text."""
    counter: Counter[tuple[bytes, ...]] = Counter()
    for token in PATTERN.findall(text):
        b = token.encode("utf-8")
        counter[tuple(bytes([c]) for c in b)] += 1
    return counter


class Tokenizer:
    def __init__(
        self,
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]],
        special_tokens: list[str] | None = None,
    ):
        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [t.encode("utf-8") for t in self.special_tokens]
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}

    def _bpe(self, token_bytes: bytes) -> list[bytes]:
        tokens = [bytes([b]) for b in token_bytes]
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i + 1]) for i in range(len(tokens) - 1)]
            ranks = [self.bpe_ranks.get(p, float("inf")) for p in pairs]
            min_rank = min(ranks)
            if min_rank == float("inf"):
                break
            idx = ranks.index(min_rank)
            tokens[idx : idx + 2] = [tokens[idx] + tokens[idx + 1]]
        return tokens

    def _encode_segment(self, text: str) -> list[int]:
        tokens = []
        for token in PATTERN.findall(text):
            for piece in self._bpe(token.encode("utf-8")):
                tokens.append(self.token_to_id[piece])
        return tokens

    def encode(self, text: str) -> list[int]:
        if not self.special_tokens:
            return self._encode_segment(text)
        tokens = []
        i = 0
        specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
        while i < len(text):
            matched = False
            for tok in specials_sorted:
                if text.startswith(tok, i):
                    tokens.append(self.token_to_id[tok.encode("utf-8")])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            j = i
            while j < len(text) and not any(
                text.startswith(st, j) for st in specials_sorted
            ):
                j += 1
            tokens.extend(self._encode_segment(text[i:j]))
            i = j
        return tokens

    def encode_iterable(self, iterable: Iterable[str]):
        for chunk in iterable:
            yield from self.encode(chunk)

    def decode(self, ids: Iterable[int]) -> str:
        byte_seq = b"".join(self.id_to_token[id] for id in ids)
        return byte_seq.decode("utf-8", errors="replace")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    *,
    profile: bool = False,
    progress: bool = False,
    use_rust: bool | None = None,
    chunk_size: int = 64 * 1024 * 1024,  # 64MB default
    force_chunked: bool = False,
    chunked_threshold: int = 100 * 1024 * 1024,  # 100MB threshold
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    profiler = cProfile.Profile() if profile else None
    if profiler:
        profiler.enable()

    if use_rust is None:
        use_rust = _rust_train_bpe is not None

    if use_rust and _rust_train_bpe:
        import os
        
        # Check file size to decide on processing method
        file_size = os.path.getsize(input_path)
        use_chunked = force_chunked or (file_size > chunked_threshold and _rust_train_bpe_from_file is not None)
        
        if use_chunked:
            if progress:
                print(f"📁 Large file detected ({file_size // (1024*1024)} MB), using chunked processing...")
            vocab_map, merges_list = _rust_train_bpe_from_file(
                input_path, vocab_size, special_tokens, progress, chunk_size
            )
        else:
            # Load entire file for smaller files
            with open(input_path, encoding="utf-8") as f:
                text = f.read()
            vocab_map, merges_list = _rust_train_bpe(
                text, vocab_size, special_tokens, progress
            )
        
        vocab = {int(k): bytes(v) for k, v in vocab_map.items()}
        merges = []
        for a_id, b_id in merges_list:
            merges.append((vocab[a_id], vocab[b_id]))
        if profiler:
            profiler.disable()
        return vocab, merges

    # Fallback to Python implementation - always load entire file
    with open(input_path, encoding="utf-8") as f:
        text = f.read()

    # Split into documents using <|endoftext|> as a delimiter when present.
    if "<|endoftext|>" in special_tokens:
        docs = text.split("<|endoftext|>")
    else:
        docs = [text]

    # Remove any other special tokens from documents before tokenizing
    other_specials = [t for t in special_tokens if t != "<|endoftext|>"]
    if other_specials:
        pattern = "(" + "|".join(re.escape(t) for t in other_specials) + ")"
        processed_docs: list[str] = []
        for doc in docs:
            parts = re.split(pattern, doc)
            processed_docs.extend(p for p in parts if p and p not in other_specials)
        docs = processed_docs

    # Collect frequency of pre-token byte sequences with multiprocessing
    word_freq: Counter[tuple[bytes, ...]] = Counter()
    num_workers = min(mp.cpu_count() or 1, len(docs))
    if num_workers > 1:
        with mp.Pool(num_workers) as pool:
            counters = list(
                tqdm(
                    pool.imap(_count_tokens, docs),
                    total=len(docs),
                    desc="Tokenizing",
                    disable=not progress,
                )
            )
        for c in counters:
            word_freq.update(c)
    else:
        for doc in tqdm(docs, desc="Tokenizing", disable=not progress):
            word_freq.update(_count_tokens(doc))

    vocab: dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    merges: list[tuple[bytes, bytes]] = []

    target_size = vocab_size - len(special_tokens)

    # precompute pair frequencies once and update them incrementally
    pair_freq: Counter[tuple[bytes, bytes]] = Counter()
    for word, freq in word_freq.items():
        for a, b in zip(word, word[1:]):
            pair_freq[(a, b)] += freq

    merges_pbar = trange(
        target_size - len(vocab), desc="BPE merges", disable=not progress
    )
    while len(vocab) < target_size and pair_freq:
        max_freq = max(pair_freq.values())
        candidates = [p for p, f in pair_freq.items() if f == max_freq]
        pair = max(candidates)  # lexicographically greatest pair

        new_token = pair[0] + pair[1]
        vocab[next_id] = new_token
        merges.append(pair)

        # remove the merged pair from frequency table
        pair_freq.pop(pair, None)

        # In-place update to avoid creating new_word_freq Counter
        words_to_update = []
        for word, freq in word_freq.items():
            # Check if word contains the pair to merge
            if any(
                word[i] == pair[0] and i + 1 < len(word) and word[i + 1] == pair[1]
                for i in range(len(word) - 1)
            ):

                # Apply merge to this word
                new_tokens = []
                i = 0
                length = len(word)
                while i < length:
                    if i < length - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(word[i])
                        i += 1

                new_word = tuple(new_tokens)
                words_to_update.append((word, new_word, freq))

                # Update pair frequencies efficiently
                # Remove old pairs
                for p in zip(word, word[1:]):
                    pair_freq[p] -= freq
                    if pair_freq[p] <= 0:
                        pair_freq.pop(p, None)

                # Add new pairs
                for p in zip(new_word, new_word[1:]):
                    pair_freq[p] = pair_freq.get(p, 0) + freq

        # Apply word updates in batch
        for old_word, new_word, freq in words_to_update:
            del word_freq[old_word]
            word_freq[new_word] = word_freq.get(new_word, 0) + freq
        next_id += 1
        merges_pbar.update(1)

    merges_pbar.close()

    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1
    if profiler:
        profiler.disable()
        s = StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats()
        print(s.getvalue())

    return vocab, merges
