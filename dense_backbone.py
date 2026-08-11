#!/usr/bin/env python3
"""
dense_backbone.py — Dense Transformer backbone operations for Swarm Layer 1.

Implements RMSNorm, RoPE (Rotary Position Embeddings), KVCache, Multi-Head
Attention, and LM Head projection for the OLMoE-1B-7B dense attention path.

Target model geometry:
  hidden_dim = 2048, num_heads = 16, num_kv_heads = 16, head_dim = 128
  vocab_size = 102848, max_seq_len = 2048

Design rules:
  - Python 3.10+, standard library + numpy only.
  - Weights may be raw ``bytes`` (from GGUF / flat file) or ``np.ndarray``.
  - No torch, no external DL frameworks.
  - Synchronous — this layer runs in a thread, not in the event loop.
"""

from __future__ import annotations

import math
import struct
from typing import Optional, Tuple, Union

import numpy as np


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _to_float32(data: Union[bytes, np.ndarray, list], expected_size: Optional[int] = None) -> np.ndarray:
    """Convert *data* to a writable float32 ``np.ndarray``.

    Accepts raw ``bytes`` (little-endian f32, as stored in GGUF/flat files),
    ``np.ndarray`` (copied and cast if needed), or ``list[float]``.

    Parameters
    ----------
    data:
        Input weight data in one of the accepted formats.
    expected_size:
        If set and *data* is ``bytes``, validate the number of float32
        elements.  ``None`` disables validation.

    Returns
    -------
    np.ndarray
        1-D float32 array.  Caller should ``.reshape()`` as needed.

    Raises
    ------
    ValueError
        If ``bytes`` size does not match *expected_size*.
    TypeError
        If *data* has an unsupported type.
    """
    if isinstance(data, np.ndarray):
        arr = data.astype(np.float32, copy=False)
        if not arr.flags["WRITEABLE"]:
            arr = arr.copy()
        return arr.ravel()
    elif isinstance(data, bytes):
        n_floats = len(data) // 4
        if expected_size is not None and n_floats != expected_size:
            raise ValueError(
                f"Expected {expected_size} float32 elements "
                f"({expected_size * 4} bytes), got {n_floats} "
                f"({len(data)} bytes)"
            )
        if len(data) % 4 != 0:
            raise ValueError(
                f"Byte length {len(data)} is not a multiple of 4"
            )
        return np.frombuffer(data, dtype=np.float32).copy()
    elif isinstance(data, list):
        return np.array(data, dtype=np.float32)
    else:
        raise TypeError(
            f"Expected bytes, np.ndarray, or list[float]; got {type(data).__name__}"
        )


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    # Clip to prevent overflow in exp.
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


# ═══════════════════════════════════════════════════════════════════════════════
# RMSNorm
# ═══════════════════════════════════════════════════════════════════════════════


class RMSNorm:
    """Root Mean Square Layer Normalisation.

    Applies pre-attention and pre-FFN normalisation:

        y = x / sqrt(mean(x²) + ε) ⊙ γ

    where *x* is the input hidden state, *γ* is a learned scale vector,
    and *ε* is a small constant for numerical stability.

    Parameters
    ----------
    weight:
        Scale parameter *γ*.  May be raw ``bytes`` (little-endian f32) or
        a ``np.ndarray`` of shape ``[hidden_dim]``.
    eps:
        Epsilon added to the denominator (default ``1e-5``).
    """

    def __init__(
        self,
        weight: Union[bytes, np.ndarray],
        eps: float = 1e-5,
    ) -> None:
        self._gamma = _to_float32(weight)
        self._eps = float(eps)

    @property
    def hidden_dim(self) -> int:
        """Dimensionality of the normalisation (length of *γ*)."""
        return self._gamma.shape[0]

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Apply RMSNorm to hidden state *x*.

        Parameters
        ----------
        x:
            Input activations, shape ``[hidden_dim]``.

        Returns
        -------
        np.ndarray
            Normalised activations, same shape as *x*.
        """
        # rms = sqrt(mean(x²) + ε)
        rms = np.sqrt(np.mean(np.square(x)) + self._eps)
        return (x / rms) * self._gamma


# ═══════════════════════════════════════════════════════════════════════════════
# RoPE — Rotary Position Embeddings
# ═══════════════════════════════════════════════════════════════════════════════


class RoPE:
    """Rotary Position Embeddings (Su et al., 2021).

    Pre-computes cosine/sine frequency tables for positions up to
    *max_seq_len* and applies the rotary transformation to query and key
    vectors at a given position.

    Parameters
    ----------
    head_dim:
        Dimension of each attention head (default ``128``).
    max_seq_len:
        Maximum sequence length for pre-computation (default ``2048``).
    theta:
        Base frequency (default ``10000.0``).
    """

    def __init__(
        self,
        head_dim: int = 128,
        max_seq_len: int = 2048,
        theta: float = 10000.0,
    ) -> None:
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len
        self._theta = theta

        # Pre-compute frequencies: θ_i = theta^(-2i/d) for i = 0, …, d/2-1
        i = np.arange(0, head_dim, 2, dtype=np.float32)
        self._freqs = 1.0 / (theta ** (i / head_dim))  # [head_dim // 2]

        # Pre-compute cos/sin for every position.
        positions = np.arange(max_seq_len, dtype=np.float32)
        # angles: [max_seq_len, head_dim // 2]
        angles = np.outer(positions, self._freqs)
        self._cos_cached = np.cos(angles).astype(np.float32)
        self._sin_cached = np.sin(angles).astype(np.float32)

    def apply(
        self,
        q: np.ndarray,
        k: np.ndarray,
        position: int,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Apply rotary position encoding to query and key vectors.

        Parameters
        ----------
        q:
            Query vectors, shape ``[num_heads, head_dim]``.
        k:
            Key vectors, shape ``[num_heads, head_dim]``.
        position:
            Absolute position index (0-based) in the sequence.

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``(q_rotated, k_rotated)``, each with the same shape as the input.
        """
        if position < 0 or position >= self._max_seq_len:
            raise ValueError(
                f"position {position} is out of range [0, {self._max_seq_len})"
            )
        cos = self._cos_cached[position]  # [head_dim // 2]
        sin = self._sin_cached[position]

        q_rot = self._rotate_half(q, cos, sin)
        k_rot = self._rotate_half(k, cos, sin)
        return q_rot, k_rot

    def _rotate_half(
        self, x: np.ndarray, cos: np.ndarray, sin: np.ndarray
    ) -> np.ndarray:
        """Apply rotary transformation to *x* using pre-computed *cos*/*sin*.

        Rotates adjacent dimension pairs: (x_0, x_1), (x_2, x_3), ...
        """
        x_rot = x.copy()
        # Even indices (0, 2, 4, ...)
        x_even = x[..., 0::2]
        # Odd indices (1, 3, 5, ...)
        x_odd = x[..., 1::2]

        x_rot[..., 0::2] = x_even * cos - x_odd * sin
        x_rot[..., 1::2] = x_even * sin + x_odd * cos
        return x_rot


# ═══════════════════════════════════════════════════════════════════════════════
# KVCache
# ═══════════════════════════════════════════════════════════════════════════════


class KVCache:
    """Per-layer Key/Value cache for autoregressive generation.

    Stores already-computed key and value vectors so they are not
    recomputed on every generation step.  Each layer has its own
    independent cache.

    Parameters
    ----------
    num_layers:
        Number of transformer layers (default ``28`` for OLMoE-1B-7B).
    num_kv_heads:
        Number of key/value attention heads (default ``16``).
    head_dim:
        Dimension of each attention head (default ``128``).
    max_seq_len:
        Maximum sequence length before cache is exhausted (default ``2048``).
    """

    def __init__(
        self,
        num_layers: int = 28,
        num_kv_heads: int = 16,
        head_dim: int = 128,
        max_seq_len: int = 2048,
    ) -> None:
        self._num_layers = num_layers
        self._num_kv_heads = num_kv_heads
        self._head_dim = head_dim
        self._max_seq_len = max_seq_len

        # Per-layer storage: each is [num_kv_heads, max_seq_len, head_dim]
        # Initialised to zero; positions beyond `_seq_len` are invalid.
        self._keys: list[np.ndarray] = [
            np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self._values: list[np.ndarray] = [
            np.zeros((num_kv_heads, max_seq_len, head_dim), dtype=np.float32)
            for _ in range(num_layers)
        ]
        self._seq_len: int = 0
        # Track which layers have been appended to in the current step
        # (before advance() is called).  get() uses this to include the
        # just-appended K/V in the returned window.
        self._appended: list[bool] = [False] * num_layers

    @property
    def seq_len(self) -> int:
        """Number of tokens currently stored in the cache."""
        return self._seq_len

    @property
    def max_seq_len(self) -> int:
        """Maximum capacity of the cache."""
        return self._max_seq_len

    def append(self, layer_idx: int, k: np.ndarray, v: np.ndarray) -> None:
        """Append one token's key and value vectors for *layer_idx*.

        Parameters
        ----------
        layer_idx:
            Which transformer layer (0-based).
        k:
            Key vectors, shape ``[num_kv_heads, head_dim]``.
        v:
            Value vectors, shape ``[num_kv_heads, head_dim]``.

        Raises
        ------
        IndexError
            If *layer_idx* is out of range.
        RuntimeError
            If the cache is full (*seq_len* == *max_seq_len*).
        """
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self._num_layers})"
            )
        if self._seq_len >= self._max_seq_len:
            raise RuntimeError(
                f"KVCache full: seq_len={self._seq_len}, "
                f"max_seq_len={self._max_seq_len}"
            )
        pos = self._seq_len
        self._keys[layer_idx][:, pos, :] = k
        self._values[layer_idx][:, pos, :] = v
        self._appended[layer_idx] = True

    def advance(self) -> None:
        """Increment the sequence length after all layers have appended.

        Must be called exactly once per generation step, after all layers
        have called ``append()`` for the current token.
        """
        if self._seq_len >= self._max_seq_len:
            raise RuntimeError("KVCache already full")
        self._seq_len += 1
        # Clear per-layer append flags for the next step.
        for i in range(self._num_layers):
            self._appended[i] = False

    def get(self, layer_idx: int) -> Tuple[np.ndarray, np.ndarray]:
        """Return ``(keys, values)`` for *layer_idx* up to current ``seq_len``.

        If ``append()`` has been called for this layer in the current step
        (before ``advance()``), the just-appended K/V is included in the
        returned window, so the attention computation can see the current
        token's own key/value during self-attention.

        Parameters
        ----------
        layer_idx:
            Which transformer layer (0-based).

        Returns
        -------
        tuple[np.ndarray, np.ndarray]
            ``keys`` and ``values``, each shape
            ``[num_kv_heads, effective_len, head_dim]`` where
            ``effective_len`` is ``seq_len + (1 if this layer was just
            appended else 0)``.
        """
        if layer_idx < 0 or layer_idx >= self._num_layers:
            raise IndexError(
                f"layer_idx {layer_idx} out of range [0, {self._num_layers})"
            )
        effective_len = self._seq_len + (
            1 if self._appended[layer_idx] else 0
        )
        return (
            self._keys[layer_idx][:, :effective_len, :].copy(),
            self._values[layer_idx][:, :effective_len, :].copy(),
        )

    def reset(self) -> None:
        """Clear all cached keys/values (e.g. for a new conversation)."""
        self._seq_len = 0
        for i in range(self._num_layers):
            self._keys[i].fill(0.0)
            self._values[i].fill(0.0)
            self._appended[i] = False


# ═══════════════════════════════════════════════════════════════════════════════
# AttentionBlock
# ═══════════════════════════════════════════════════════════════════════════════


class AttentionBlock:
    """Multi-Head Attention block for one dense transformer layer.

    Computes Q, K, V linear projections, applies RoPE, reads/writes the
    KV cache, and runs scaled dot-product attention.

    Parameters
    ----------
    w_q:
        Query projection weight.  Raw ``bytes`` or ``np.ndarray`` of shape
        ``[hidden_dim, hidden_dim]`` (= ``[2048, 2048]``).
    w_k:
        Key projection weight, same shape.
    w_v:
        Value projection weight, same shape.
    w_o:
        Output projection weight, same shape.
    num_heads:
        Number of attention heads (default ``16``).
    head_dim:
        Dimension per head (default ``128``).
    """

    def __init__(
        self,
        w_q: Union[bytes, np.ndarray],
        w_k: Union[bytes, np.ndarray],
        w_v: Union[bytes, np.ndarray],
        w_o: Union[bytes, np.ndarray],
        num_heads: int = 16,
        head_dim: int = 128,
    ) -> None:
        hidden_dim = num_heads * head_dim
        self._num_heads = num_heads
        self._head_dim = head_dim
        self._hidden_dim = hidden_dim

        # Reshape to [hidden_dim, hidden_dim] for matrix multiply.
        self._W_q = _to_float32(w_q, hidden_dim * hidden_dim).reshape(
            hidden_dim, hidden_dim
        )
        self._W_k = _to_float32(w_k, hidden_dim * hidden_dim).reshape(
            hidden_dim, hidden_dim
        )
        self._W_v = _to_float32(w_v, hidden_dim * hidden_dim).reshape(
            hidden_dim, hidden_dim
        )
        self._W_o = _to_float32(w_o, hidden_dim * hidden_dim).reshape(
            hidden_dim, hidden_dim
        )

    @property
    def num_heads(self) -> int:
        return self._num_heads

    @property
    def head_dim(self) -> int:
        return self._head_dim

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    def forward(
        self,
        x: np.ndarray,
        rope: RoPE,
        kv_cache: KVCache,
        layer_idx: int,
        position: int,
    ) -> np.ndarray:
        """Run one attention pass for a single token.

        Parameters
        ----------
        x:
            Input hidden state, shape ``[hidden_dim]``.
        rope:
            Pre-initialised ``RoPE`` instance.
        kv_cache:
            The fleet-wide ``KVCache`` for all layers.
        layer_idx:
            Which layer this attention block belongs to (0-based).
        position:
            Absolute token position in the sequence (0-based).

        Returns
        -------
        np.ndarray
            Attention output, shape ``[hidden_dim]``.
        """
        # ── 1. Linear projections ─────────────────────────────────────
        q = self._W_q @ x  # [hidden_dim]
        k = self._W_k @ x
        v = self._W_v @ x

        # ── 2. Reshape to [num_heads, head_dim] ───────────────────────
        q = q.reshape(self._num_heads, self._head_dim)
        k = k.reshape(self._num_heads, self._head_dim)
        v = v.reshape(self._num_heads, self._head_dim)

        # ── 3. Apply RoPE ─────────────────────────────────────────────
        q, k = rope.apply(q, k, position)

        # ── 4. Append to KV cache ─────────────────────────────────────
        kv_cache.append(layer_idx, k, v)

        # ── 5. Retrieve full cached K, V ──────────────────────────────
        K_full, V_full = kv_cache.get(layer_idx)
        # K_full: [num_heads, seq_len, head_dim]
        # V_full: [num_heads, seq_len, head_dim]
        seq_len = K_full.shape[1]

        # ── 6. Scaled dot-product attention ───────────────────────────
        # scores[h, i] = sum_d Q[h, d] * K[h, i, d]
        scores = (
            np.einsum("hd,hnd->hn", q, K_full).astype(np.float32)
            / math.sqrt(self._head_dim)
        )

        # ── 7. Causal mask (optional during generation — all cached
        #       positions are ≤ current position by construction) ──────
        # During prefill we would apply a triangular mask.  For
        # autoregressive generation the cache only contains valid
        # positions, so no mask is needed.

        # ── 8. Softmax ───────────────────────────────────────────────
        attn_weights = _softmax(scores, axis=-1)  # [num_heads, seq_len]

        # ── 9. Weighted sum of values ─────────────────────────────────
        # out[h, d] = sum_i attn_weights[h, i] * V[h, i, d]
        out_heads = np.einsum("hn,hnd->hd", attn_weights, V_full).astype(np.float32)

        # ── 10. Concatenate heads and project ─────────────────────────
        out_concat = out_heads.reshape(self._hidden_dim)  # [hidden_dim]
        output = self._W_o @ out_concat  # [hidden_dim]
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# LMHead
# ═══════════════════════════════════════════════════════════════════════════════


class LMHead:
    """Language Model head — projects final hidden state to vocabulary logits.

    Uses the ``output.weight`` tensor from the model, shape
    ``[vocab_size, hidden_dim]`` (= ``[102848, 2048]``).

    Parameters
    ----------
    weight:
        Output projection weight.  Raw ``bytes`` or ``np.ndarray`` of shape
        ``[vocab_size, hidden_dim]``.
    """

    def __init__(
        self,
        weight: Union[bytes, np.ndarray],
        vocab_size: int = 102848,
        hidden_dim: int = 2048,
    ) -> None:
        self._vocab_size = vocab_size
        self._hidden_dim = hidden_dim
        self._W = _to_float32(weight, vocab_size * hidden_dim).reshape(
            vocab_size, hidden_dim
        )

    @property
    def vocab_size(self) -> int:
        return self._vocab_size

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Project hidden state *x* to vocabulary logits.

        Parameters
        ----------
        x:
            Final-layer hidden state, shape ``[hidden_dim]``.

        Returns
        -------
        np.ndarray
            Logits for every vocabulary token, shape ``[vocab_size]``.
        """
        return self._W @ x


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════


def _self_test() -> None:
    """Quick smoke-test of each component."""
    rng = np.random.RandomState(42)

    print("── dense_backbone self-test ──")

    # ── RMSNorm ─────────────────────────────────────────────────────
    hidden_dim = 2048
    gamma = np.ones(hidden_dim, dtype=np.float32)
    rms = RMSNorm(gamma, eps=1e-5)
    x = rng.randn(hidden_dim).astype(np.float32)
    y = rms.forward(x)
    # RMS of output should be ~1.0
    rms_out = np.sqrt(np.mean(np.square(y)))
    assert abs(rms_out - 1.0) < 0.1, f"RMSNorm output RMS={rms_out:.3f}, expected ~1.0"
    print("  RMSNorm: ✓")

    # Test with bytes input
    rms_bytes = RMSNorm(gamma.tobytes(), eps=1e-5)
    y2 = rms_bytes.forward(x)
    assert np.allclose(y, y2, atol=1e-6)
    print("  RMSNorm (bytes weight): ✓")

    # ── RoPE ─────────────────────────────────────────────────────────
    rope = RoPE(head_dim=128, max_seq_len=2048, theta=10000.0)
    q = rng.randn(16, 128).astype(np.float32)
    k = rng.randn(16, 128).astype(np.float32)
    q_rot, k_rot = rope.apply(q, k, 0)
    assert q_rot.shape == (16, 128)
    assert k_rot.shape == (16, 128)
    # Rotation should preserve per-head vector norms.
    for h in range(16):
        norm_q = np.linalg.norm(q[h])
        norm_q_rot = np.linalg.norm(q_rot[h])
        assert abs(norm_q - norm_q_rot) < 1e-4, (
            f"Head {h}: q norm changed {norm_q:.4f} → {norm_q_rot:.4f}"
        )
        norm_k = np.linalg.norm(k[h])
        norm_k_rot = np.linalg.norm(k_rot[h])
        assert abs(norm_k - norm_k_rot) < 1e-4, (
            f"Head {h}: k norm changed {norm_k:.4f} → {norm_k_rot:.4f}"
        )
    # Different positions should give different rotations.
    q_rot_p1, _ = rope.apply(q, k, 1)
    assert not np.allclose(q_rot, q_rot_p1, atol=1e-6), (
        "Position 0 and 1 should produce different rotations"
    )
    print("  RoPE: ✓")

    # ── KVCache ──────────────────────────────────────────────────────
    cache = KVCache(num_layers=2, num_kv_heads=16, head_dim=128, max_seq_len=2048)
    assert cache.seq_len == 0

    k1 = rng.randn(16, 128).astype(np.float32)
    v1 = rng.randn(16, 128).astype(np.float32)
    cache.append(0, k1, v1)
    cache.append(1, k1, v1)
    assert cache.seq_len == 0  # Not advanced yet
    cache.advance()
    assert cache.seq_len == 1

    K, V = cache.get(0)
    assert K.shape == (16, 1, 128)
    assert np.allclose(K[:, 0, :], k1, atol=1e-6)
    assert V.shape == (16, 1, 128)
    print("  KVCache: ✓")

    # ── AttentionBlock ───────────────────────────────────────────────
    # Use identity matrices for Q and O so outputs are predictable.
    eye = np.eye(hidden_dim, dtype=np.float32)
    attn = AttentionBlock(
        w_q=eye,
        w_k=eye,
        w_v=eye,
        w_o=eye,
        num_heads=16,
        head_dim=128,
    )
    cache2 = KVCache(num_layers=1, num_kv_heads=16, head_dim=128, max_seq_len=2048)
    rope2 = RoPE(head_dim=128, max_seq_len=2048)

    x_in = rng.randn(hidden_dim).astype(np.float32)
    out = attn.forward(x_in, rope2, cache2, 0, 0)
    cache2.advance()
    assert out.shape == (hidden_dim,), f"Expected ({hidden_dim},), got {out.shape}"
    # With identity Q/K/V/O and a single token, RoPE at position 0 leaves
    # vectors unchanged (cos=1, sin=0), so Q=K=V=x_in, softmax of a
    # 1-element score is 1.0, and attention returns the value = x_in.
    # W_o is also identity, so output ≈ input.  This is correct behaviour.
    assert np.allclose(out, x_in, atol=1e-3), (
        "With identity weights and single token, output should equal input"
    )
    print("  AttentionBlock: ✓")

    # ── LMHead ───────────────────────────────────────────────────────
    # Small test with reduced vocab for speed.
    vocab_small = 1024
    w_lm = rng.randn(vocab_small, hidden_dim).astype(np.float32)
    lm = LMHead(w_lm, vocab_size=vocab_small, hidden_dim=hidden_dim)
    logits = lm.forward(x_in)
    assert logits.shape == (vocab_small,)
    # Verify with bytes input
    lm_bytes = LMHead(w_lm.tobytes(), vocab_size=vocab_small, hidden_dim=hidden_dim)
    logits2 = lm_bytes.forward(x_in)
    assert np.allclose(logits, logits2, atol=1e-5)
    print("  LMHead: ✓")

    print("── dense_backbone self-test passed ──")


if __name__ == "__main__":
    _self_test()
