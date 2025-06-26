from __future__ import annotations
import regex as re
from collections import Counter
from typing import Iterable, List, Tuple, Dict, Optional

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


def train_bpe(input_path: str, vocab_size: int, special_tokens: List[str]) -> Tuple[Dict[int, bytes], List[Tuple[bytes, bytes]]]:
    with open(input_path, 'r', encoding='utf-8') as f:
        text = f.read()
    if special_tokens:
        pattern = '(' + '|'.join(re.escape(t) for t in special_tokens) + ')'
        parts = re.split(pattern, text)
        text_parts = [p for p in parts if p and p not in special_tokens]
    else:
        text_parts = [text]
    word_freq: Counter[Tuple[int, ...]] = Counter()
    for part in text_parts:
        for token in PATTERN.findall(part):
            b = token.encode('utf-8')
            word_freq[tuple(b)] += 1
    vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
    next_id = 256
    merges: List[Tuple[bytes, bytes]] = []
    while len(vocab) < vocab_size - len(special_tokens):
        pair_freq: Counter[Tuple[int, int]] = Counter()
        for word, freq in word_freq.items():
            for a, b in zip(word, word[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        max_freq = max(pair_freq.values())
        candidates = [p for p, f in pair_freq.items() if f == max_freq]
        pair = max(candidates)
        new_token = vocab[pair[0]] + vocab[pair[1]]
        vocab[next_id] = new_token
        merges.append((vocab[pair[0]], vocab[pair[1]]))
        new_word_freq: Counter[Tuple[int, ...]] = Counter()
        for word, freq in word_freq.items():
            i = 0
            new_word = []
            w = list(word)
            while i < len(w):
                if i < len(w)-1 and w[i] == pair[0] and w[i+1] == pair[1]:
                    new_word.append(next_id)
                    i += 2
                else:
                    new_word.append(w[i])
                    i += 1
            new_word_freq[tuple(new_word)] += freq
        word_freq = new_word_freq
        next_id += 1
    for st in special_tokens:
        vocab[next_id] = st.encode('utf-8')
        next_id += 1
    return vocab, merges
