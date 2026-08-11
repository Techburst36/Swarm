#!/usr/bin/env python3
"""
tokenizer.py — BPE tokenizer for the Swarm distributed inference engine.

Attempts to load the ``tokenizers`` package (HuggingFace) for real models.
Falls back to parsing GGUF metadata for standard byte-level BPE
tokenization/detokenization when the package is unavailable.

Design rules:
  - Python 3.10+, standard library only for the fallback path.
  - Exposes ``encode(text) -> list[int]`` and ``decode(tokens) -> str``.
"""

from __future__ import annotations

import os
import re
from typing import Any


class SwarmTokenizer:
    """Byte-level BPE tokenizer backed by GGUF metadata or HuggingFace.

    Parameters
    ----------
    model_path:
        Path to a GGUF file with tokenizer metadata, or a HuggingFace
        model ID (e.g. ``"allenai/OLMoE-1B-7B-0924"``).  GGUF files are
        detected by extension or by the ``GGUF`` magic at byte 0.
    vocab:
        Direct vocabulary list (for testing, bypasses GGUF/HF loading).
    merges:
        Direct BPE merge pairs (for testing).
    """

    def __init__(
        self,
        model_path: str = "",
        *,
        vocab: list[str] | None = None,
        merges: list[tuple[str, str]] | None = None,
        bos_token_id: int | None = None,
        eos_token_id: int | None = None,
        pad_token_id: int | None = None,
        unk_token_id: int | None = None,
    ) -> None:
        self._model_path = model_path
        self._vocab: list[str] = []
        self._vocab_map: dict[str, int] = {}
        self._merges: list[tuple[str, str]] = []

        self._bos_token_id: int | None = bos_token_id
        self._eos_token_id: int | None = eos_token_id
        self._pad_token_id: int | None = pad_token_id
        self._unk_token_id: int | None = unk_token_id

        self._hf_tokenizer: Any = None

        # Direct construction from vocab list (for testing).
        if vocab is not None:
            self._vocab = list(vocab)
            self._vocab_map = {t: i for i, t in enumerate(self._vocab)}
            if merges is not None:
                self._merges = list(merges)
            return

        # Try HuggingFace first, then GGUF fallback.
        if model_path and self._try_huggingface():
            return
        if model_path:
            self._load_from_gguf()

    # ── Public API ──────────────────────────────────────────────────────

    def encode(self, text: str) -> list[int]:
        """Tokenize *text* into a list of integer token IDs."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.encode(text).ids

        return self._bpe_encode(text)

    def decode(self, tokens: list[int]) -> str:
        """Decode a list of token IDs back to text."""
        if self._hf_tokenizer is not None:
            return self._hf_tokenizer.decode(tokens)

        return self._bpe_decode(tokens)

    @property
    def vocab_size(self) -> int:
        """Number of tokens in the vocabulary."""
        return len(self._vocab)

    @property
    def bos_token_id(self) -> int | None:
        """Beginning-of-sequence token ID."""
        return self._bos_token_id

    @property
    def eos_token_id(self) -> int | None:
        """End-of-sequence token ID."""
        return self._eos_token_id

    @property
    def pad_token_id(self) -> int | None:
        """Padding token ID."""
        return self._pad_token_id

    @property
    def unk_token_id(self) -> int | None:
        """Unknown token ID (defaults to 0 if not set)."""
        return self._unk_token_id

    @property
    def vocab(self) -> list[str]:
        """The full vocabulary as a list of token strings."""
        return list(self._vocab)

    # ── HuggingFace path ────────────────────────────────────────────────

    def _try_huggingface(self) -> bool:
        """Attempt to load the model via HuggingFace tokenizers."""
        try:
            from tokenizers import Tokenizer  # type: ignore[import-untyped]
        except ImportError:
            return False

        model_id = self._model_path
        if model_id.endswith(".gguf") and os.path.isfile(model_id):
            try:
                from gguf_stream_reader import GGUFReader
                reader = GGUFReader(model_id)
                hf_name = reader.get_metadata("general.name")
                if hf_name:
                    model_id = hf_name
            except Exception:
                pass

        try:
            self._hf_tokenizer = Tokenizer.from_pretrained(model_id)
        except Exception:
            return False

        bos = (self._hf_tokenizer.token_to_id("<s>")
               or self._hf_tokenizer.token_to_id("<|begin_of_text|>"))
        eos = (self._hf_tokenizer.token_to_id("</s>")
               or self._hf_tokenizer.token_to_id("<|end_of_text|>"))
        pad = self._hf_tokenizer.token_to_id("<pad>")
        unk = self._hf_tokenizer.token_to_id("<unk>")

        self._bos_token_id = bos
        self._eos_token_id = eos
        self._pad_token_id = pad
        self._unk_token_id = unk
        return True

    # ── GGUF fallback ───────────────────────────────────────────────────

    def _load_from_gguf(self) -> None:
        """Parse tokenizer metadata from a GGUF file."""
        if not os.path.isfile(self._model_path):
            raise FileNotFoundError(
                f"GGUF file not found: {self._model_path}"
            )

        from gguf_stream_reader import GGUFReader

        reader = GGUFReader(self._model_path)

        tokens_array = reader.get_metadata("tokenizer.ggml.tokens")
        if tokens_array is None:
            raise ValueError(
                "GGUF file has no 'tokenizer.ggml.tokens' metadata — "
                "cannot build tokenizer from this file."
            )
        self._vocab = list(tokens_array)
        self._vocab_map = {t: i for i, t in enumerate(self._vocab)}

        self._bos_token_id = reader.get_metadata(
            "tokenizer.ggml.bos_token_id", None
        )
        self._eos_token_id = reader.get_metadata(
            "tokenizer.ggml.eos_token_id", None
        )
        self._pad_token_id = reader.get_metadata(
            "tokenizer.ggml.padding_token_id", None
        )
        self._unk_token_id = reader.get_metadata(
            "tokenizer.ggml.unknown_token_id", None
        )

        merges_array = reader.get_metadata("tokenizer.ggml.merges")
        if merges_array is not None:
            for merge_str in merges_array:
                parts = merge_str.split(" ", 1)
                if len(parts) == 2:
                    self._merges.append((parts[0], parts[1]))

        added = reader.get_metadata("tokenizer.ggml.added_tokens")
        if added is not None:
            for added_str in added:
                if added_str not in self._vocab_map:
                    self._vocab_map[added_str] = len(self._vocab)
                    self._vocab.append(added_str)

    # ── BPE encoding ────────────────────────────────────────────────────

    # Pre-tokenization: split on whitespace (simple fallback).
    _WHITESPACE_RE = re.compile(r"\S+|\s+")

    def _bpe_encode(self, text: str) -> list[int]:
        """Encode *text* using byte-level BPE from GGUF metadata.

        If no merge table is present (simple character vocabulary), falls
        back to direct character-to-token lookup.
        """
        if not self._merges:
            return self._simple_encode(text)

        token_ids: list[int] = []
        for match in self._WHITESPACE_RE.finditer(text):
            word = match.group()
            if not word:
                continue
            token_ids.extend(self._bpe_encode_word(word))

        return token_ids

    def _simple_encode(self, text: str) -> list[int]:
        """Encode without BPE merges — direct character-to-token lookup."""
        token_ids: list[int] = []
        for ch in text:
            tid = self._vocab_map.get(ch)
            if tid is not None:
                token_ids.append(tid)
            else:
                # Fall back to unk token.
                token_ids.append(self._unk_token_id or 0)
        return token_ids

    def _bpe_encode_word(self, word: str) -> list[int]:
        """Apply BPE merges to a single word (pre-tokenized chunk)."""
        # Convert to UTF-8 bytes, represent each byte as a string.
        byte_reprs = []
        for byte in word.encode("utf-8"):
            # Try raw byte character first, then hex representation.
            byte_char = bytes([byte]).decode("latin-1")
            if byte_char in self._vocab_map:
                byte_reprs.append(byte_char)
            else:
                hex_str = f"<0x{byte:02X}>"
                if hex_str in self._vocab_map:
                    byte_reprs.append(hex_str)
                else:
                    byte_reprs.append(byte_char)

        # Build merge rank dictionary.
        merge_rank: dict[tuple[str, str], int] = {}
        for rank, (a, b) in enumerate(self._merges):
            merge_rank[(a, b)] = rank

        # Apply merges greedily.
        while len(byte_reprs) > 1:
            best_rank = None
            best_idx = -1
            for i in range(len(byte_reprs) - 1):
                pair = (byte_reprs[i], byte_reprs[i + 1])
                rank = merge_rank.get(pair)
                if rank is not None and (best_rank is None or rank < best_rank):
                    best_rank = rank
                    best_idx = i
            if best_rank is None:
                break
            merged = byte_reprs[best_idx] + byte_reprs[best_idx + 1]
            byte_reprs = (
                byte_reprs[:best_idx] + [merged] + byte_reprs[best_idx + 2:]
            )

        # Convert to token IDs.
        token_ids: list[int] = []
        for token_str in byte_reprs:
            tid = self._vocab_map.get(token_str)
            if tid is None:
                tid = self._unk_token_id or 0
            token_ids.append(tid)
        return token_ids

    # ── BPE decoding ────────────────────────────────────────────────────

    def _bpe_decode(self, tokens: list[int]) -> str:
        """Decode token IDs back to text via vocabulary lookup."""
        parts: list[str] = []
        for tid in tokens:
            if 0 <= tid < len(self._vocab):
                parts.append(self._vocab[tid])
            else:
                parts.append("")
        text = "".join(parts)
        return text


# ── Simple character tokenizer factory (for testing) ────────────────────


def make_char_tokenizer(
    vocab_chars: str = "",
    *,
    bos_token: str = "<s>",
    eos_token: str = "</s>",
    unk_token: str = "<unk>",
    pad_token: str = "<pad>",
) -> SwarmTokenizer:
    """Build a SwarmTokenizer with a minimal character vocabulary.

    Uses direct construction — no GGUF file needed.  For testing only.

    Parameters
    ----------
    vocab_chars:
        String of characters to include in the vocabulary.  Defaults to
        printable ASCII (32–126).
    """
    if not vocab_chars:
        vocab_chars = "".join(chr(i) for i in range(32, 127))

    tokens = [bos_token, eos_token, unk_token, pad_token]
    tokens.extend(list(vocab_chars))

    return SwarmTokenizer(
        vocab=tokens,
        merges=[],
        bos_token_id=0,
        eos_token_id=1,
        unk_token_id=2,
        pad_token_id=3,
    )


# ── Self-test ─────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Verify the tokenizer works with a synthetic vocabulary."""
    print("── tokenizer self-test ──")

    tok = make_char_tokenizer()
    print(f"  Vocab size: {tok.vocab_size}")
    print(f"  BOS: {tok.bos_token_id}, EOS: {tok.eos_token_id}")

    test_str = "Hello, World!"
    encoded = tok.encode(test_str)
    decoded = tok.decode(encoded)
    print(f"  '{test_str}' -> {encoded}")
    print(f"  -> '{decoded}'")

    assert decoded == test_str, f"Round-trip failed: '{test_str}' != '{decoded}'"
    print("  Round-trip: ✓")

    assert tok.vocab_size > 10, "Vocab too small"
    print("── tokenizer self-test passed ──")


if __name__ == "__main__":
    _self_test()
