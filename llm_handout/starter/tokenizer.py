# """Baseline tokenizer: raw UTF-8 bytes, vocab of 256. Simple, never fails on
# unseen text — and treats a Devanagari character as 3 tokens. Think about
# what that does to your model's context window and your token budget on the
# Hindi part of the corpus.

# You may replace this with anything you train ON THE PROVIDED CORPUS ONLY
# (e.g., BPE), as long as:
#   1. it can encode ARBITRARY UTF-8 text (byte-level fallback) and it is
#      LOSSLESS: decode(encode(text)) == text, exactly. The scorer and the
#      graders both verify this round-trip — a lossy tokenizer makes bpb
#      meaningless and disqualifies the run.
#   2. this file keeps exposing:  load() -> tokenizer object with
#      .encode(str) -> list[int], .decode(list[int]) -> str, .vocab_size.
#      train.py and evaluate.py call load() with NO arguments — keep any
#      extra parameters optional.
#   3. anything it needs is saved under your submission folder and loaded by
#      load() with no internet. Grading runs with cwd = your folder; resolve
#      saved files relative to __file__ to be safe.
# """
# """
# Custom BPE Tokenizer. 
# Run this file standalone once to train it: `python tokenizer.py`
# """
# import json
# import os
# from collections import Counter

# class BPETokenizer:
#     def __init__(self, vocab_size=1024):
#         self.vocab_size = vocab_size
#         self.merges = {} # (int, int) -> int
#         self.vocab = {i: bytes([i]) for i in range(256)}
        
#     def _encode_chunk(self, tokens):
#         # Helper function that encodes a small block of tokens
#         while len(tokens) > 1:
#             # Find all adjacent pairs
#             pairs = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
#             # Find the pair that can be merged earliest
#             pair_to_merge = min(pairs, key=lambda p: self.merges.get(p, float('inf')))
#             if pair_to_merge not in self.merges:
#                 break 
            
#             # Replace occurrences of pair_to_merge
#             new_id = self.merges[pair_to_merge]
#             new_tokens = []
#             i = 0
#             while i < len(tokens):
#                 if i < len(tokens) - 1 and (tokens[i], tokens[i+1]) == pair_to_merge:
#                     new_tokens.append(new_id)
#                     i += 2
#                 else:
#                     new_tokens.append(tokens[i])
#                     i += 1
#             tokens = new_tokens
#         return tokens

#     def encode(self, text):
#         # Convert entire text to raw bytes once
#         raw_bytes = list(text.encode("utf-8"))
#         encoded = []
        
#         # Process in chunks of 500 bytes to avoid O(N^2) slowness
#         chunk_size = 500
#         for i in range(0, len(raw_bytes), chunk_size):
#             chunk = raw_bytes[i : i + chunk_size]
#             encoded.extend(self._encode_chunk(chunk))
            
#         return encoded

#     def decode(self, ids):
#         b = bytearray()
#         for idx in ids:
#             b.extend(self.vocab[idx])
#         return b.decode("utf-8", errors="replace")

#     def save(self, path):
#         # Convert tuple keys to strings for JSON
#         str_merges = {f"{k[0]},{k[1]}": v for k, v in self.merges.items()}
#         with open(path, "w") as f:
#             json.dump({"merges": str_merges, "vocab_size": self.vocab_size}, f)

#     def load_from_file(self, path):
#         with open(path, "r") as f:
#             data =json.loads(f.read())
#         self.vocab_size = data["vocab_size"]
#         self.merges = {tuple(map(int, k.split(","))): v for k, v in data["merges"].items()}
        
#         # Rebuild vocab mapping
#         self.vocab = {i: bytes([i]) for i in range(256)}
#         for (p0, p1), idx in self.merges.items():
#             self.vocab[idx] = self.vocab[p0] + self.vocab[p1]

# def load():
#     """Called by train.py and evaluate.py"""
#     tok = BPETokenizer()
#     merge_file = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
#     if os.path.exists(merge_file):
#         tok.load_from_file(merge_file)
#     else:
#         # Fallback for initial tests before training
#         print("WARNING: bpe_merges.json not found. Falling back to ByteTokenizer behavior.")
#         tok.vocab_size = 256
#     return tok

# if __name__ == "__main__":
#     # BPE Training Script
#     print("Training BPE Tokenizer...")
#     corpus_path = os.path.join(os.path.dirname(__file__), "..", "data", "train_corpus.txt")
#     with open(corpus_path, "r", encoding="utf-8") as f:
#         text = f.read()
    
#     tokens = list(text.encode("utf-8"))
#     target_vocab = 1024
#     num_merges = target_vocab - 256
    
#     tok = BPETokenizer(vocab_size=target_vocab)
    
#     for i in range(num_merges):
#         # Count frequency of adjacent pairs
#         pairs = Counter(zip(tokens, tokens[1:]))
#         if not pairs:
#             break
#         best_pair = pairs.most_common(1)[0][0]
#         new_id = 256 + i
        
#         tok.merges[best_pair] = new_id
#         tok.vocab[new_id] = tok.vocab[best_pair[0]] + tok.vocab[best_pair[1]]
        
#         # Apply merge to the token sequence
#         new_tokens = []
#         j = 0
#         while j < len(tokens):
#             if j < len(tokens) - 1 and (tokens[j], tokens[j+1]) == best_pair:
#                 new_tokens.append(new_id)
#                 j += 2
#             else:
#                 new_tokens.append(tokens[j])
#                 j += 1
#         tokens = new_tokens
        
#         if (i+1) % 100 == 0:
#             print(f"Merge {i+1}/{num_merges} completed. Unique tokens length: {len(tokens)}")

#     save_path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
#     tok.save(save_path)
#     print(f"Saved to {save_path}")

"""Byte-level BPE tokenizer trained ONLY on train_corpus.txt.
Falls back to raw bytes for anything unmerged -> always lossless on
arbitrary UTF-8 (required by evaluate.py's round-trip check).
"""
import json
import re
import os

_SPLIT = re.compile(r"\s+|\S+")  # partitions text into whitespace/non-whitespace runs


class BPETokenizer:
    def __init__(self, merges=None):
        # merges: ordered list of [a, b, new_id] as learned (rank = list order)
        self.merges = merges or []
        self.rank = {(a, b): i for i, (a, b, _) in enumerate(self.merges)}
        self.pair2id = {(a, b): nid for a, b, nid in self.merges}
        self.vocab_size = 256 + len(self.merges)
        # id -> raw bytes, for O(1) decode
        self.id2bytes = [bytes([i]) for i in range(256)]
        for a, b, nid in self.merges:
            self.id2bytes.append(self.id2bytes[a] + self.id2bytes[b])

    # ---- training (incremental pair-count updates: only touch words that
    # actually contain the merged pair each step, not the whole corpus) ----
    @classmethod
    def train(cls, text, num_merges):
        from collections import Counter, defaultdict
        chunks = _SPLIT.findall(text)
        freq = Counter(tuple(c.encode("utf-8")) for c in chunks)  # word -> count

        pair_counts = Counter()
        pair_to_words = defaultdict(set)
        for w, c in freq.items():
            for i in range(len(w) - 1):
                p = (w[i], w[i + 1])
                pair_counts[p] += c
                pair_to_words[p].add(w)

        merges = []
        for _ in range(num_merges):
            if not pair_counts:
                break
            best = max(pair_counts, key=pair_counts.get)
            new_id = 256 + len(merges)
            for w in list(pair_to_words.get(best, ())):
                c = freq.pop(w, None)
                if c is None:
                    continue
                for i in range(len(w) - 1):
                    p = (w[i], w[i + 1])
                    pair_counts[p] -= c
                    if pair_counts[p] <= 0:
                        del pair_counts[p]
                    pair_to_words[p].discard(w)
                out, i = [], 0
                while i < len(w):
                    if i < len(w) - 1 and (w[i], w[i + 1]) == best:
                        out.append(new_id)
                        i += 2
                    else:
                        out.append(w[i])
                        i += 1
                nw = tuple(out)
                freq[nw] = freq.get(nw, 0) + c
                for i in range(len(nw) - 1):
                    p = (nw[i], nw[i + 1])
                    pair_counts[p] = pair_counts.get(p, 0) + c
                    pair_to_words[p].add(nw)
            pair_to_words.pop(best, None)
            merges.append([best[0], best[1], new_id])
        return cls(merges)

    # ---- inference ----
    def _encode_chunk(self, b):
        ids = list(b)
        while len(ids) > 1:
            pairs = [(self.rank.get((ids[i], ids[i + 1]), None), i)
                     for i in range(len(ids) - 1)]
            pairs = [p for p in pairs if p[0] is not None]
            if not pairs:
                break
            _, i = min(pairs, key=lambda x: x[0])
            a, bb = ids[i], ids[i + 1]
            ids = ids[:i] + [self.pair2id[(a, bb)]] + ids[i + 2:]
        return ids

    def encode(self, text):
        out = []
        for chunk in _SPLIT.findall(text):
            out.extend(self._encode_chunk(chunk.encode("utf-8")))
        return out

    def decode(self, ids):
        return b"".join(self.id2bytes[i] for i in ids).decode("utf-8", errors="replace")

    def save(self, path):
        with open(path, "w") as f:
            json.dump({"merges": self.merges}, f)


def load(path=None):
    """Loads bpe_merges.json next to this file. Falls back to pure-byte
    (vocab 256) tokenizer if no merges file is present, so this never fails."""
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "bpe_merges.json")
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        return BPETokenizer(data["merges"])
    return BPETokenizer(merges=[])


if __name__ == "__main__":
    import argparse, time
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--vocab", type=int, default=2048)
    ap.add_argument("--out", default="bpe_merges.json")
    args = ap.parse_args()
    text = open(args.data, encoding="utf-8").read()
    t0 = time.time()
    tok = BPETokenizer.train(text, args.vocab - 256)
    tok.save(args.out)
    print(f"trained vocab={tok.vocab_size} merges={len(tok.merges)} "
          f"in {time.time()-t0:.0f}s -> {args.out}")
    # sanity: round-trip check
    sample = text[:200000]
    assert tok.decode(tok.encode(sample)) == sample, "round-trip failed!"
    print("round-trip OK")