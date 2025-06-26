from __future__ import annotations

import cProfile
import pstats
import regex as re
from collections import Counter
from io import StringIO
from typing import Dict, Iterable, List, Optional, Tuple

PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\v\p{L}\p{N}]+|\s+(?!\S)|\s+")


class Tokenizer:
    def __init__(self, vocab: Dict[int, bytes], merges: List[Tuple[bytes, bytes]], special_tokens: Optional[List[str]] = None):
        self.id_to_token = dict(vocab)
        self.token_to_id = {v: k for k, v in vocab.items()}
        self.special_tokens = special_tokens or []
        self.special_token_bytes = [t.encode('utf-8') for t in self.special_tokens]
        self.bpe_ranks = {merge: i for i, merge in enumerate(merges)}

    def _bpe(self, token_bytes: bytes) -> List[bytes]:
        tokens = [bytes([b]) for b in token_bytes]
        while len(tokens) >= 2:
            pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            ranks = [self.bpe_ranks.get(p, float('inf')) for p in pairs]
            min_rank = min(ranks)
            if min_rank == float('inf'):
                break
            idx = ranks.index(min_rank)
            tokens[idx:idx+2] = [tokens[idx] + tokens[idx+1]]
        return tokens

    def _encode_segment(self, text: str) -> List[int]:
        tokens = []
        for token in PATTERN.findall(text):
            for piece in self._bpe(token.encode('utf-8')):
                tokens.append(self.token_to_id[piece])
        return tokens

    def encode(self, text: str) -> List[int]:
        if not self.special_tokens:
            return self._encode_segment(text)
        tokens = []
        i = 0
        specials_sorted = sorted(self.special_tokens, key=len, reverse=True)
        while i < len(text):
            matched = False
            for tok in specials_sorted:
                if text.startswith(tok, i):
                    tokens.append(self.token_to_id[tok.encode('utf-8')])
                    i += len(tok)
                    matched = True
                    break
            if matched:
                continue
            j = i
            while j < len(text) and not any(text.startswith(st, j) for st in specials_sorted):
                j += 1
            tokens.extend(self._encode_segment(text[i:j]))
            i = j
        return tokens

    def encode_iterable(self, iterable: Iterable[str]):
        for chunk in iterable:
            for token in self.encode(chunk):
                yield token

    def decode(self, ids: Iterable[int]) -> str:
        byte_seq = b"".join(self.id_to_token[id] for id in ids)
        return byte_seq.decode('utf-8', errors='replace')


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: List[str],
    *,
    profile: bool = False,
) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    profiler = cProfile.Profile() if profile else None
    if profiler:
        profiler.enable()

    with open(input_path, "r", encoding="utf-8") as f:
        text = f.read()

    if special_tokens:
        pattern = "(" + "|".join(re.escape(t) for t in special_tokens) + ")"
        parts = re.split(pattern, text)
        text_parts = [p for p in parts if p and p not in special_tokens]
    else:
        text_parts = [text]

    # Collect frequency of pre-token byte sequences
    word_freq: Counter[Tuple[bytes, ...]] = Counter()
    for part in text_parts:
        for token in PATTERN.findall(part):
            b = token.encode("utf-8")
            word_freq[tuple(bytes([c]) for c in b)] += 1

    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    merges: List[Tuple[bytes, bytes]] = []

    target_size = vocab_size - len(special_tokens)
    while len(vocab) < target_size:
        pair_freq: Counter[Tuple[bytes, bytes]] = Counter()
        for word, freq in word_freq.items():
            tokens = list(word)
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break

        max_freq = max(pair_freq.values())
        candidates = [p for p, f in pair_freq.items() if f == max_freq]
        pair = max(candidates)  # lexicographically greatest pair

        new_token = pair[0] + pair[1]
        vocab[next_id] = new_token
        merges.append(pair)

        new_word_freq: Counter[Tuple[bytes, ...]] = Counter()
        for word, freq in word_freq.items():
            tokens = list(word)
            i = 0
            new_tokens = []
            while i < len(tokens):
                if i < len(tokens) - 1 and tokens[i] == pair[0] and tokens[i + 1] == pair[1]:
                    new_tokens.append(new_token)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_word_freq[tuple(new_tokens)] += freq
        word_freq = new_word_freq
        next_id += 1

    for st in special_tokens:
        vocab[next_id] = st.encode("utf-8")
        next_id += 1
    if profiler:
        profiler.disable()
        s = StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats()
        print(s.getvalue())

    return vocab, merges
