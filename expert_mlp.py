#!/usr/bin/env python3
"""
expert_mlp.py — SwiGLU Expert MLP Engine for Swarm Layer 1.

Evaluates a single MoE expert via 3-pass SwiGLU using the existing
``inference_core.execute_expert`` SIMD GEMM, plus the Router that
selects which experts fire.

Target geometry (OLMoE-1B-7B):
  hidden_dim = 2048, num_experts = 64, top_k = 8
  expert intermediate_dim = 1024

SwiGLU activation for expert *e*:

    Expert_e(x) = W_down ⸨ Swish(W_gate · x) ⊙ (W_up · x) ⸩

where Swish(z) = z · σ(z).

Three GEMM passes (each via ``inference_core.execute_expert``):
  1. gate = W_gate @ x     [1024, 2048] @ [2048] → [1024]
  2. up   = W_up   @ x     [1024, 2048] @ [2048] → [1024]
  3. out  = W_down @ gated [2048, 1024] @ [1024] → [2048]

Supports F32 and Q4_K_M quantized weight buffers.

Design rules:
  - Python 3.10+, standard library + numpy + existing inference_core.
  - Weights as raw ``bytes`` (GGUF / flat file compatible).
  - Synchronous — runs in a thread, not in the event loop.
  - No torch, no external DL frameworks.
"""

from __future__ import annotations

import math
from typing import List, Optional, Tuple, Union

import numpy as np

from inference_core import execute_expert


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════


def _sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise sigmoid."""
    z = np.clip(z, -20.0, 20.0)
    return 1.0 / (1.0 + np.exp(-z))


def _softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax along *axis*."""
    x_max = np.max(x, axis=axis, keepdims=True)
    e_x = np.exp(x - x_max)
    return e_x / np.sum(e_x, axis=axis, keepdims=True)


def _as_list_of_floats(x: Union[np.ndarray, List[float]]) -> List[float]:
    """Convert *x* to ``list[float]`` for ``execute_expert``."""
    if isinstance(x, np.ndarray):
        return x.ravel().tolist()
    return list(x)


def _validate_weight_bytes(
    data: bytes, rows: int, cols: int, quant_type: str, label: str
) -> None:
    """Check that *data* is large enough for the declared shape and quant."""
    if quant_type == "f32":
        expected = rows * cols * 4
    elif quant_type == "q4_k_m":
        blocks_per_row = (cols + 255) // 256
        expected = rows * blocks_per_row * 144  # Q4K_BLOCK_BYTES
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")
    if len(data) < expected:
        raise ValueError(
            f"{label}: expected ≥ {expected} bytes for [{rows}, {cols}] "
            f"{quant_type}, got {len(data)} bytes"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# SwiGLUExpert
# ═══════════════════════════════════════════════════════════════════════════════


class SwiGLUExpert:
    """One SwiGLU MoE expert evaluated via three GEMM passes.

    Weights are stored as raw ``bytes`` and dispatched to
    ``inference_core.execute_expert``, which selects the C AVX2 backend or
    the pure-Python fallback at call time.

    Parameters
    ----------
    w_gate:
        Gate projection weight ``bytes``, shape ``[intermediate_dim, hidden_dim]``
        (``[1024, 2048]`` for OLMoE-1B-7B).
    w_up:
        Up projection weight ``bytes``, same shape as *w_gate*.
    w_down:
        Down projection weight ``bytes``, shape ``[hidden_dim, intermediate_dim]``
        (``[2048, 1024]``).
    hidden_dim:
        Model hidden dimension (default ``2048``).
    intermediate_dim:
        Expert FFN intermediate dimension (default ``1024``).
    quant_type:
        ``"f32"`` for float32 or ``"q4_k_m"`` for Q4_K_M block-quantized
        weights (default ``"f32"``).
    """

    def __init__(
        self,
        w_gate: bytes,
        w_up: bytes,
        w_down: bytes,
        hidden_dim: int = 2048,
        intermediate_dim: int = 1024,
        quant_type: str = "f32",
    ) -> None:
        self._hidden_dim = hidden_dim
        self._intermediate_dim = intermediate_dim
        self._quant_type = quant_type

        _validate_weight_bytes(
            w_gate, intermediate_dim, hidden_dim, quant_type, "w_gate"
        )
        _validate_weight_bytes(
            w_up, intermediate_dim, hidden_dim, quant_type, "w_up"
        )
        _validate_weight_bytes(
            w_down, hidden_dim, intermediate_dim, quant_type, "w_down"
        )

        self._w_gate = w_gate
        self._w_up = w_up
        self._w_down = w_down

    @property
    def hidden_dim(self) -> int:
        return self._hidden_dim

    @property
    def intermediate_dim(self) -> int:
        return self._intermediate_dim

    @property
    def quant_type(self) -> str:
        return self._quant_type

    def forward(self, x: Union[np.ndarray, List[float]]) -> np.ndarray:
        """Evaluate the expert on input activations *x*.

        Three GEMM passes:
          1. gate = W_gate @ x
          2. up   = W_up   @ x
          3. out  = W_down @ (Swish(gate) ⊙ up)

        Parameters
        ----------
        x:
            Input activation vector, length ``hidden_dim``.

        Returns
        -------
        np.ndarray
            Output activation vector, shape ``[hidden_dim]``.
        """
        x_list = _as_list_of_floats(x)
        d = self._hidden_dim

        if len(x_list) != d:
            raise ValueError(
                f"Input length {len(x_list)} != hidden_dim {d}"
            )

        # ── Pass 1: gate = W_gate @ x ─────────────────────────────────
        gate_raw = execute_expert(
            self._w_gate, x_list, self._intermediate_dim, d, self._quant_type
        )
        gate = np.array(gate_raw, dtype=np.float32)

        # ── Pass 2: up = W_up @ x ─────────────────────────────────────
        up_raw = execute_expert(
            self._w_up, x_list, self._intermediate_dim, d, self._quant_type
        )
        up = np.array(up_raw, dtype=np.float32)

        # ── Swish(gate) ⊙ up ─────────────────────────────────────────
        swish_gate = gate * _sigmoid(gate)
        gated = swish_gate * up  # elementwise

        # ── Pass 3: out = W_down @ gated ─────────────────────────────
        gated_list: List[float] = gated.tolist()
        out_raw = execute_expert(
            self._w_down,
            gated_list,
            d,  # rows = hidden_dim
            self._intermediate_dim,  # cols = intermediate_dim
            self._quant_type,
        )
        return np.array(out_raw, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Router
# ═══════════════════════════════════════════════════════════════════════════


class Router:
    """MoE Router — selects top-*k* experts for a given hidden state.

    Computes ``logits = W_router @ x``, selects the *top_k* entries,
    and returns their indices and Softmax-normalised probabilities.

    Parameters
    ----------
    weight:
        Router weight matrix.  Raw ``bytes`` or ``np.ndarray`` of shape
        ``[num_experts, hidden_dim]`` (``[64, 2048]`` for OLMoE-1B-7B).
    num_experts:
        Total number of experts (default ``64``).
    top_k:
        Number of experts to route to (default ``8``).
    hidden_dim:
        Model hidden dimension (default ``2048``).
    """

    def __init__(
        self,
        weight: Union[bytes, np.ndarray],
        num_experts: int = 64,
        top_k: int = 8,
        hidden_dim: int = 2048,
    ) -> None:
        self._num_experts = num_experts
        self._top_k = top_k
        self._hidden_dim = hidden_dim

        # Accept both bytes and ndarray.
        if isinstance(weight, np.ndarray):
            self._W = weight.astype(np.float32, copy=False).reshape(
                num_experts, hidden_dim
            )
        elif isinstance(weight, bytes):
            expected = num_experts * hidden_dim
            n_floats = len(weight) // 4
            if n_floats < expected:
                raise ValueError(
                    f"Router weight: expected ≥ {expected} floats "
                    f"({expected * 4} bytes), got {n_floats} "
                    f"({len(weight)} bytes)"
                )
            self._W = np.frombuffer(weight, dtype=np.float32).copy().reshape(
                num_experts, hidden_dim
            )
        else:
            raise TypeError(
                f"Expected bytes or np.ndarray, got {type(weight).__name__}"
            )

        if top_k > num_experts:
            raise ValueError(
                f"top_k ({top_k}) must be ≤ num_experts ({num_experts})"
            )

    @property
    def num_experts(self) -> int:
        return self._num_experts

    @property
    def top_k(self) -> int:
        return self._top_k

    def forward(
        self, x: Union[np.ndarray, List[float]]
    ) -> Tuple[List[int], np.ndarray]:
        """Route hidden state *x* to top-*k* experts.

        Parameters
        ----------
        x:
            Input hidden state, length ``hidden_dim``.

        Returns
        -------
        tuple[list[int], np.ndarray]
            - ``indices``: list of *top_k* expert indices (0-based),
              sorted descending by routing weight.
            - ``probs``: ``np.ndarray`` of length *top_k* with Softmax
              probabilities over the selected logits.  Always sums to 1.0.
        """
        if isinstance(x, list):
            x = np.array(x, dtype=np.float32)
        x = x.ravel().astype(np.float32)

        if x.shape[0] != self._hidden_dim:
            raise ValueError(
                f"Input length {x.shape[0]} != hidden_dim {self._hidden_dim}"
            )

        # ── Compute router logits ─────────────────────────────────────
        logits = self._W @ x  # [num_experts]

        # ── Top-k selection ───────────────────────────────────────────
        # argpartition is O(n) rather than O(n log n) for argsort;
        # only the top k need ordering.
        if self._top_k == self._num_experts:
            indices = list(range(self._num_experts))
            top_logits = logits
            # Sort by descending logit.
            order = np.argsort(-logits)
            indices = [indices[i] for i in order]
            top_logits = logits[order]
        else:
            part_indices = np.argpartition(-logits, self._top_k)[: self._top_k]
            top_logits = logits[part_indices]
            # Sort the top-k by descending logit.
            sorted_local = np.argsort(-top_logits)
            indices = part_indices[sorted_local].tolist()
            top_logits = top_logits[sorted_local]

        # ── Softmax over selected logits ──────────────────────────────
        probs = _softmax(top_logits)
        return indices, probs


# ═══════════════════════════════════════════════════════════════════════════════
# Self-test
# ═══════════════════════════════════════════════════════════════════════════════


def _self_test() -> None:
    """Quick smoke-test of Router and SwiGLUExpert."""
    import struct

    rng = np.random.RandomState(42)

    print("── expert_mlp self-test ──")

    hidden_dim = 2048
    intermediate_dim = 1024
    num_experts = 64
    top_k = 8

    # ── Router ───────────────────────────────────────────────────────
    w_router = rng.randn(num_experts, hidden_dim).astype(np.float32)
    router = Router(w_router, num_experts=num_experts, top_k=top_k, hidden_dim=hidden_dim)

    x = rng.randn(hidden_dim).astype(np.float32)
    indices, probs = router.forward(x)

    assert len(indices) == top_k, f"Expected {top_k} indices, got {len(indices)}"
    assert len(probs) == top_k
    assert abs(float(np.sum(probs)) - 1.0) < 1e-5, (
        f"Probabilities sum to {np.sum(probs):.6f}, expected 1.0"
    )
    # Indices must be in range and unique.
    assert all(0 <= i < num_experts for i in indices), (
        f"Index out of range: {indices}"
    )
    assert len(set(indices)) == top_k, f"Duplicate indices: {indices}"
    # Sorted descending by probability.
    for i in range(len(probs) - 1):
        assert probs[i] >= probs[i + 1] - 1e-6, (
            f"Probabilities not sorted descending: {probs}"
        )
    print(f"  Router: top-{top_k} indices={indices[:4]}…, sum(probs)={np.sum(probs):.4f} ✓")

    # Test bytes input.
    router_bytes = Router(w_router.tobytes(), num_experts=num_experts, top_k=top_k, hidden_dim=hidden_dim)
    indices2, probs2 = router_bytes.forward(x)
    assert indices == indices2
    assert np.allclose(probs, probs2, atol=1e-5)
    print("  Router (bytes weight): ✓")

    # ── SwiGLUExpert ─────────────────────────────────────────────────
    # Use identity-like weights so the output is predictable:
    #   W_gate = scaled identity → gate = scale * x[:1024]
    #   W_up   = scaled identity → up   = scale * x[:1024]
    #   W_down = identity         → out[:1024] = gated, out[1024:] = 0
    # (since W_down maps from 1024→2048, identity is [2048,1024],
    #  which pads with rows of zeros)

    scale = 0.25
    # W_gate: [1024, 2048] — identity in the first 1024 cols
    w_gate_mat = np.zeros((intermediate_dim, hidden_dim), dtype=np.float32)
    w_gate_mat[:, :intermediate_dim] = np.eye(intermediate_dim, dtype=np.float32) * scale
    w_gate_bytes = w_gate_mat.tobytes()

    # W_up: same pattern
    w_up_mat = np.zeros((intermediate_dim, hidden_dim), dtype=np.float32)
    w_up_mat[:, :intermediate_dim] = np.eye(intermediate_dim, dtype=np.float32) * scale
    w_up_bytes = w_up_mat.tobytes()

    # W_down: [2048, 1024] — identity in first 1024 rows
    w_down_mat = np.zeros((hidden_dim, intermediate_dim), dtype=np.float32)
    w_down_mat[:intermediate_dim, :] = np.eye(intermediate_dim, dtype=np.float32)
    w_down_bytes = w_down_mat.tobytes()

    expert = SwiGLUExpert(
        w_gate=w_gate_bytes,
        w_up=w_up_bytes,
        w_down=w_down_bytes,
        hidden_dim=hidden_dim,
        intermediate_dim=intermediate_dim,
        quant_type="f32",
    )

    # Input: first 1024 dims = 2.0, rest = 0.0
    x_expert = np.zeros(hidden_dim, dtype=np.float32)
    x_expert[:intermediate_dim] = 2.0

    out = expert.forward(x_expert)
    assert out.shape == (hidden_dim,), (
        f"Expected ({hidden_dim},), got {out.shape}"
    )

    # Expected:
    #   gate = scale * 2.0 = 0.5  (for first 1024 dims)
    #   swish(0.5) = 0.5 * sigmoid(0.5) ≈ 0.5 * 0.6225 = 0.3112
    #   up = 0.5  (same as gate since weights are identical)
    #   gated = 0.3112 * 0.5 = 0.1556
    #   out[:1024] = gated, out[1024:] = 0

    gate_val = scale * 2.0
    swish_val = gate_val * (1.0 / (1.0 + math.exp(-gate_val)))
    gated_val = swish_val * gate_val

    # First 1024 outputs should be ~gated_val
    assert np.allclose(out[:intermediate_dim], gated_val, atol=1e-3), (
        f"First dim: expected ~{gated_val:.4f}, got {out[0]:.4f}"
    )
    # Last 1024 should be ~0
    assert np.allclose(out[intermediate_dim:], 0.0, atol=1e-4), (
        f"Last dim: expected 0, got {out[intermediate_dim]:.6f}"
    )
    print(f"  SwiGLUExpert: output[:4]={out[:4].tolist()}, output[-4:]={out[-4:].tolist()} ✓")

    # ── Verify backward compatibility: list input ────────────────────
    out2 = expert.forward(x_expert.tolist())
    assert np.allclose(out, out2, atol=1e-5)
    print("  SwiGLUExpert (list input): ✓")

    print("── expert_mlp self-test passed ──")


if __name__ == "__main__":
    _self_test()
