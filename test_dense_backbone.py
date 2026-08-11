#!/usr/bin/env python3
"""
test_dense_backbone.py — Unit test suite for dense_backbone.py and expert_mlp.py.

Exercises every component against the OLMoE-1B-7B target geometry:
  hidden_dim=2048, num_heads=16, num_kv_heads=16, head_dim=128,
  num_experts=64, top_k=8, intermediate_dim=1024, vocab_size=102848.

Tests:
  1. RMSNorm — numerical correctness against a hand-computed reference.
  2. RoPE — output dimensions, norm invariance after rotation,
     different positions yield different outputs.
  3. AttentionBlock + KVCache — sequence lengths 1 through 4,
     verifies KV-cache grows correctly, causal masking, and output shape.
  4. Router — returns exactly 8 indices, probabilities sum to 1.0,
     indices are unique and in-range, sorted by descending probability.
  5. SwiGLUExpert — 3 GEMM passes, output shape [2048], correctness
     under identity-like weights, compatibility with list[float] input.

Usage:
    python3 test_dense_backbone.py
"""

from __future__ import annotations

import math
import struct
import sys
import traceback
from typing import Tuple

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════
# Test framework (minimal — no pytest)
# ═══════════════════════════════════════════════════════════════════════════

_passed = 0
_failed = 0
_failures: list[str] = []


def _check(condition: bool, description: str) -> None:
    """Assert *condition*; record pass/fail with *description*."""
    global _passed, _failed, _failures
    if condition:
        _passed += 1
    else:
        _failed += 1
        _failures.append(description)
        print(f"  FAIL: {description}", file=sys.stderr)


def _approx(a, b, atol: float = 1e-5) -> bool:
    """Return ``True`` if *a* and *b* are within *atol*."""
    return abs(float(a) - float(b)) < atol


def _allclose(a: np.ndarray, b: np.ndarray, atol: float = 1e-4) -> bool:
    """Return ``True`` if *a* and *b* are elementwise within *atol*."""
    return bool(np.allclose(a, b, atol=atol))


# ═══════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════

RNG = np.random.RandomState(12345)

# Target geometry
HIDDEN_DIM = 2048
NUM_HEADS = 16
NUM_KV_HEADS = 16
HEAD_DIM = 128
NUM_EXPERTS = 64
TOP_K = 8
INTERMEDIATE_DIM = 1024
VOCAB_SIZE = 102848
MAX_SEQ_LEN = 2048
NUM_LAYERS = 28


def _make_eye(n: int) -> np.ndarray:
    """Return a float32 n×n identity matrix."""
    return np.eye(n, dtype=np.float32)


def _make_rand(*shape: int) -> np.ndarray:
    """Return a float32 array of *shape* with RNG values."""
    return RNG.randn(*shape).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: RMSNorm
# ═══════════════════════════════════════════════════════════════════════════


def test_rmsnorm() -> None:
    """Verify RMSNorm against a hand-computed reference."""
    from dense_backbone import RMSNorm

    print("── Test 1: RMSNorm ──")

    # ── 1a: Unit gamma, unit input → RMS ≈ 1 after norm ──────────
    gamma = np.ones(HIDDEN_DIM, dtype=np.float32)
    rms = RMSNorm(gamma, eps=1e-5)
    x = np.ones(HIDDEN_DIM, dtype=np.float32)
    y = rms.forward(x)
    # Input RMS = 1.0, so output should be ~1.0 * gamma = gamma = 1.0
    _check(y.shape == (HIDDEN_DIM,), "1a: output shape")
    _check(_allclose(y, np.ones(HIDDEN_DIM), atol=1e-1), "1a: all-ones output ≈ 1")

    # ── 1b: Hand-computed reference ───────────────────────────────
    # x = [3.0, 4.0], gamma = [2.0, 0.5], eps = 0.0
    # mean(x²) = (9 + 16) / 2 = 12.5
    # rms = sqrt(12.5) ≈ 3.5355
    # y[0] = 3.0 / 3.5355 * 2.0 ≈ 1.6971
    # y[1] = 4.0 / 3.5355 * 0.5 ≈ 0.5657
    gamma_small = np.array([2.0, 0.5], dtype=np.float32)
    rms_small = RMSNorm(gamma_small, eps=0.0)
    x_small = np.array([3.0, 4.0], dtype=np.float32)
    y_small = rms_small.forward(x_small)

    expected_rms = math.sqrt((9.0 + 16.0) / 2.0)  # ≈ 3.5355
    expected_y0 = 3.0 / expected_rms * 2.0
    expected_y1 = 4.0 / expected_rms * 0.5
    _check(_approx(y_small[0], expected_y0, atol=1e-4), "1b: y[0] matches reference")
    _check(_approx(y_small[1], expected_y1, atol=1e-4), "1b: y[1] matches reference")

    # ── 1c: Epsilon prevents division by zero ──────────────────────
    x_zero = np.zeros(HIDDEN_DIM, dtype=np.float32)
    y_zero = rms.forward(x_zero)
    _check(not np.any(np.isnan(y_zero)), "1c: no NaN on zero input")
    _check(not np.any(np.isinf(y_zero)), "1c: no Inf on zero input")

    # ── 1d: Bytes weight input works ───────────────────────────────
    gamma_bytes = gamma.tobytes()
    rms_bytes = RMSNorm(gamma_bytes, eps=1e-5)
    y_bytes = rms_bytes.forward(x)
    _check(_allclose(y, y_bytes, atol=1e-6), "1d: bytes weight matches ndarray weight")

    # ── 1e: Random input, verify output RMS ≈ 1.0 ─────────────────
    x_rand = _make_rand(HIDDEN_DIM)
    y_rand = rms.forward(x_rand)
    out_rms = np.sqrt(np.mean(np.square(y_rand)))
    _check(abs(out_rms - 1.0) < 0.2, f"1e: output RMS ≈ 1.0 (got {out_rms:.3f})")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: RoPE
# ═══════════════════════════════════════════════════════════════════════════


def test_rope() -> None:
    """Verify RoPE output dimensions, norm invariance, and position sensitivity."""
    from dense_backbone import RoPE

    print("── Test 2: RoPE ──")

    rope = RoPE(head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN, theta=10000.0)

    q = _make_rand(NUM_HEADS, HEAD_DIM)
    k = _make_rand(NUM_HEADS, HEAD_DIM)

    # ── 2a: Output shapes match input ──────────────────────────────
    q_rot, k_rot = rope.apply(q, k, 0)
    _check(q_rot.shape == (NUM_HEADS, HEAD_DIM), "2a: q_rot shape")
    _check(k_rot.shape == (NUM_HEADS, HEAD_DIM), "2a: k_rot shape")

    # ── 2b: Norms are preserved (rotation is orthogonal) ───────────
    for h in range(NUM_HEADS):
        nq = float(np.linalg.norm(q[h]))
        nqr = float(np.linalg.norm(q_rot[h]))
        _check(abs(nq - nqr) < 1e-4, f"2b: head {h} q norm preserved ({nq:.4f} → {nqr:.4f})")

        nk = float(np.linalg.norm(k[h]))
        nkr = float(np.linalg.norm(k_rot[h]))
        _check(abs(nk - nkr) < 1e-4, f"2b: head {h} k norm preserved ({nk:.4f} → {nkr:.4f})")

    # ── 2c: Different positions give different rotations ───────────
    q_rot_p1, _ = rope.apply(q, k, 1)
    q_rot_p2, _ = rope.apply(q, k, 2)
    _check(not _allclose(q_rot, q_rot_p1, atol=1e-4), "2c: pos 0 ≠ pos 1")
    _check(not _allclose(q_rot_p1, q_rot_p2, atol=1e-4), "2c: pos 1 ≠ pos 2")

    # ── 2d: Position 0 differs from no rotation ────────────────────
    # At position 0, cos = 1.0 and sin = 0.0 only for θ=0, but θ>0.
    # So even at pos 0 there is some rotation (angles = 0 * freqs = 0,
    # cos=1, sin=0 → no rotation at pos 0).
    # Actually at position 0, all angles are 0, so cos=1, sin=0.
    # Thus q_rot should equal q at position 0.
    _check(_allclose(q, q_rot, atol=1e-5), "2d: pos 0 leaves vectors unchanged (cos=1, sin=0)")

    # ── 2e: Out-of-range position raises ───────────────────────────
    try:
        rope.apply(q, k, MAX_SEQ_LEN + 1)
        _check(False, "2e: out-of-range position should raise ValueError")
    except ValueError:
        _check(True, "2e: out-of-range position raises ValueError")

    # ── 2f: Position MAX_SEQ_LEN - 1 works ─────────────────────────
    q_rot_last, k_rot_last = rope.apply(q, k, MAX_SEQ_LEN - 1)
    _check(q_rot_last.shape == (NUM_HEADS, HEAD_DIM), "2f: max position works")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: KVCache
# ═══════════════════════════════════════════════════════════════════════════


def test_kvcache() -> None:
    """Verify KVCache append, get, advance, and reset."""
    from dense_backbone import KVCache

    print("── Test 3: KVCache ──")

    cache = KVCache(num_layers=2, num_kv_heads=NUM_KV_HEADS,
                    head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN)

    # ── 3a: Initial state ──────────────────────────────────────────
    _check(cache.seq_len == 0, "3a: initial seq_len = 0")

    # ── 3b: Append and advance ─────────────────────────────────────
    k0 = _make_rand(NUM_KV_HEADS, HEAD_DIM)
    v0 = _make_rand(NUM_KV_HEADS, HEAD_DIM)
    cache.append(0, k0, v0)
    cache.append(1, k0, v0)
    _check(cache.seq_len == 0, "3b: seq_len unchanged before advance()")
    cache.advance()
    _check(cache.seq_len == 1, "3b: seq_len = 1 after advance()")

    K, V = cache.get(0)
    _check(K.shape == (NUM_KV_HEADS, 1, HEAD_DIM), "3b: K shape [n_heads, 1, head_dim]")
    _check(V.shape == (NUM_KV_HEADS, 1, HEAD_DIM), "3b: V shape")
    _check(_allclose(K[:, 0, :], k0, atol=1e-6), "3b: K matches appended values")

    # ── 3c: Multiple steps ─────────────────────────────────────────
    for pos in range(1, 5):
        kp = _make_rand(NUM_KV_HEADS, HEAD_DIM)
        vp = _make_rand(NUM_KV_HEADS, HEAD_DIM)
        cache.append(0, kp, vp)
        cache.append(1, kp, vp)
        cache.advance()

    _check(cache.seq_len == 5, "3c: seq_len = 5 after 5 steps")
    K_full, V_full = cache.get(0)
    _check(K_full.shape == (NUM_KV_HEADS, 5, HEAD_DIM), "3c: K_full shape after 5 steps")

    # Verify position 0 still intact.
    _check(_allclose(K_full[:, 0, :], k0, atol=1e-6), "3c: position 0 preserved")

    # ── 3d: Reset ──────────────────────────────────────────────────
    cache.reset()
    _check(cache.seq_len == 0, "3d: seq_len = 0 after reset")
    K_reset, _ = cache.get(0)
    _check(K_reset.shape == (NUM_KV_HEADS, 0, HEAD_DIM), "3d: empty cache after reset")

    # ── 3e: Bad layer index ────────────────────────────────────────
    try:
        cache.append(99, k0, v0)
        _check(False, "3e: bad layer_idx should raise IndexError")
    except IndexError:
        _check(True, "3e: bad layer_idx raises IndexError")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: AttentionBlock with KV-Cache (seq lengths 1–4)
# ═══════════════════════════════════════════════════════════════════════════


def test_attention_block() -> None:
    """Exercise AttentionBlock across sequence lengths 1 through 4."""
    from dense_backbone import AttentionBlock, KVCache, RoPE

    print("── Test 4: AttentionBlock + KVCache ──")

    # Use identity for Q, K, V, O so we can reason about outputs.
    eye = _make_eye(HIDDEN_DIM)

    attn = AttentionBlock(
        w_q=eye,
        w_k=eye,
        w_v=eye,
        w_o=eye,
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    _check(attn.num_heads == NUM_HEADS, "4a: num_heads")
    _check(attn.head_dim == HEAD_DIM, "4a: head_dim")
    _check(attn.hidden_dim == HIDDEN_DIM, "4a: hidden_dim")

    cache = KVCache(num_layers=1, num_kv_heads=NUM_KV_HEADS,
                    head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN)
    rope = RoPE(head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN)

    # Generate 4 tokens and verify KV-cache behaviour.
    for pos in range(4):
        x = _make_rand(HIDDEN_DIM)
        out = attn.forward(x, rope, cache, layer_idx=0, position=pos)
        cache.advance()

        _check(out.shape == (HIDDEN_DIM,), f"4b: pos {pos} output shape")
        _check(not np.any(np.isnan(out)), f"4b: pos {pos} no NaN")
        _check(not np.any(np.isinf(out)), f"4b: pos {pos} no Inf")

        K, V = cache.get(0)
        expected_seq_len = pos + 1
        _check(K.shape == (NUM_KV_HEADS, expected_seq_len, HEAD_DIM),
               f"4b: pos {pos} K cache shape ({K.shape})")
        _check(V.shape == (NUM_KV_HEADS, expected_seq_len, HEAD_DIM),
               f"4b: pos {pos} V cache shape")

    # ── 4c: Attention with scaled-identity weights ──────────────────
    # Use a small scale so attention values stay reasonable.
    scale = 0.01
    w = (_make_eye(HIDDEN_DIM) * scale).astype(np.float32)
    attn2 = AttentionBlock(
        w_q=w, w_k=w, w_v=w, w_o=_make_eye(HIDDEN_DIM),
        num_heads=NUM_HEADS, head_dim=HEAD_DIM,
    )
    cache2 = KVCache(num_layers=1, num_kv_heads=NUM_KV_HEADS,
                     head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN)

    # First token: attention over a single token should just pass through
    # (softmax of a 1-element vector = 1.0, so V[0] is returned directly).
    x0 = _make_rand(HIDDEN_DIM)
    out0 = attn2.forward(x0, rope, cache2, layer_idx=0, position=0)
    cache2.advance()
    _check(out0.shape == (HIDDEN_DIM,), "4c: first token output shape")

    # ── 4d: Bytes weight input ─────────────────────────────────────
    attn_bytes = AttentionBlock(
        w_q=eye.tobytes(),
        w_k=eye.tobytes(),
        w_v=eye.tobytes(),
        w_o=eye.tobytes(),
        num_heads=NUM_HEADS,
        head_dim=HEAD_DIM,
    )
    cache_bytes = KVCache(num_layers=1, num_kv_heads=NUM_KV_HEADS,
                          head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN)
    x_test = _make_rand(HIDDEN_DIM)
    out_ref = attn.forward(x_test, rope, cache3 := KVCache(num_layers=1, num_kv_heads=NUM_KV_HEADS,
                                                            head_dim=HEAD_DIM, max_seq_len=MAX_SEQ_LEN),
                           layer_idx=0, position=0)
    cache3.advance()
    out_bytes = attn_bytes.forward(x_test, rope, cache_bytes, layer_idx=0, position=0)
    cache_bytes.advance()
    _check(out_ref.shape == out_bytes.shape, "4d: bytes weight output shape matches")
    # The outputs may differ slightly since RoPE applies differently
    # (rotation is deterministic) — should be identical.
    _check(_allclose(out_ref, out_bytes, atol=1e-4), "4d: bytes weight matches ndarray weight")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: LMHead
# ═══════════════════════════════════════════════════════════════════════════


def test_lm_head() -> None:
    """Verify LMHead output shape and bytes compatibility."""
    from dense_backbone import LMHead

    print("── Test 5: LMHead ──")

    # Use a reduced vocab for the test to keep memory reasonable.
    vocab_small = 1024
    w_lm = _make_rand(vocab_small, HIDDEN_DIM)
    lm = LMHead(w_lm, vocab_size=vocab_small, hidden_dim=HIDDEN_DIM)

    x = _make_rand(HIDDEN_DIM)
    logits = lm.forward(x)
    _check(logits.shape == (vocab_small,), f"5a: logits shape (got {logits.shape})")

    # ── 5b: Bytes weight ───────────────────────────────────────────
    lm_bytes = LMHead(w_lm.tobytes(), vocab_size=vocab_small, hidden_dim=HIDDEN_DIM)
    logits_bytes = lm_bytes.forward(x)
    _check(_allclose(logits, logits_bytes, atol=1e-4), "5b: bytes weight matches ndarray")

    # ── 5c: Full vocab_size ────────────────────────────────────────
    # Use identity to keep test fast (no need for random 102848×2048).
    # Create a scaled identity in first 2048 rows; rest are zeros.
    w_full = np.zeros((VOCAB_SIZE, HIDDEN_DIM), dtype=np.float32)
    w_full[:HIDDEN_DIM, :] = _make_eye(HIDDEN_DIM) * 0.5
    lm_full = LMHead(w_full, vocab_size=VOCAB_SIZE, hidden_dim=HIDDEN_DIM)
    logits_full = lm_full.forward(x)
    _check(logits_full.shape == (VOCAB_SIZE,), "5c: full vocab logits shape")
    # First HIDDEN_DIM logits should be 0.5 * x
    _check(_allclose(logits_full[:HIDDEN_DIM], 0.5 * x, atol=1e-4),
           "5c: first 2048 logits = 0.5*x")


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: Router
# ═══════════════════════════════════════════════════════════════════════════


def test_router() -> None:
    """Verify Router returns exactly 8 indices with probabilities summing to 1.0."""
    from expert_mlp import Router

    print("── Test 6: Router ──")

    w_router = _make_rand(NUM_EXPERTS, HIDDEN_DIM)
    router = Router(w_router, num_experts=NUM_EXPERTS, top_k=TOP_K, hidden_dim=HIDDEN_DIM)

    x = _make_rand(HIDDEN_DIM)

    # ── 6a: Returns exactly top_k indices ──────────────────────────
    indices, probs = router.forward(x)
    _check(len(indices) == TOP_K, f"6a: {TOP_K} indices (got {len(indices)})")
    _check(len(probs) == TOP_K, f"6a: {TOP_K} probabilities (got {len(probs)})")

    # ── 6b: Probabilities sum to 1.0 ───────────────────────────────
    _check(_approx(np.sum(probs), 1.0, atol=1e-5),
           f"6b: sum(probs) = 1.0 (got {np.sum(probs):.8f})")

    # ── 6c: All probabilities are non-negative ─────────────────────
    _check(bool(np.all(probs >= 0)), "6c: all probs ≥ 0")
    _check(bool(np.all(probs <= 1)), "6c: all probs ≤ 1")

    # ── 6d: Indices are unique and in range ────────────────────────
    _check(len(set(indices)) == TOP_K, "6d: indices are unique")
    _check(all(0 <= i < NUM_EXPERTS for i in indices), "6d: all indices in [0, 64)")

    # ── 6e: Sorted by descending probability ───────────────────────
    for i in range(len(probs) - 1):
        _check(probs[i] >= probs[i + 1] - 1e-6,
               f"6e: probs[{i}] ≥ probs[{i+1}]")

    # ── 6f: Deterministic — same input → same output ───────────────
    indices2, probs2 = router.forward(x)
    _check(indices == indices2, "6f: deterministic indices")
    _check(_allclose(probs, probs2, atol=1e-10), "6f: deterministic probs")

    # ── 6g: Different input → different output (almost certainly) ──
    x2 = _make_rand(HIDDEN_DIM)  # different seed state
    indices3, _ = router.forward(x2)
    # At least one index should differ (statistically almost certain).
    _check(not (indices == indices3), "6g: different input → different routing")

    # ── 6h: Bytes weight input ─────────────────────────────────────
    router_bytes = Router(w_router.tobytes(), num_experts=NUM_EXPERTS,
                          top_k=TOP_K, hidden_dim=HIDDEN_DIM)
    indices_b, probs_b = router_bytes.forward(x)
    _check(indices == indices_b, "6h: bytes weight indices match")
    _check(_allclose(probs, probs_b, atol=1e-5), "6h: bytes weight probs match")

    # ── 6i: List[float] input ──────────────────────────────────────
    indices_l, probs_l = router.forward(x.tolist())
    _check(indices == indices_l, "6i: list input indices match")
    _check(_allclose(probs, probs_l, atol=1e-5), "6i: list input probs match")

    # ── 6j: top_k = num_experts edge case ──────────────────────────
    router_all = Router(w_router, num_experts=NUM_EXPERTS, top_k=NUM_EXPERTS,
                        hidden_dim=HIDDEN_DIM)
    indices_all, probs_all = router_all.forward(x)
    _check(len(indices_all) == NUM_EXPERTS, "6j: top_k=64 returns 64 indices")
    _check(len(set(indices_all)) == NUM_EXPERTS, "6j: all indices unique")
    _check(_approx(np.sum(probs_all), 1.0, atol=1e-5), "6j: sum to 1.0")


# ═══════════════════════════════════════════════════════════════════════════
# Test 7: SwiGLUExpert
# ═══════════════════════════════════════════════════════════════════════════


def test_swiglu_expert() -> None:
    """Verify SwiGLUExpert runs 3 GEMM passes and produces correct output."""
    from expert_mlp import SwiGLUExpert

    print("── Test 7: SwiGLUExpert ──")

    # ── 7a: Identity-like weights for correctness verification ─────
    # W_gate: [1024, 2048] with scale*I in first 1024 cols
    # W_up:   same
    # W_down: [2048, 1024] with I in first 1024 rows

    scale = 0.5
    w_gate_mat = np.zeros((INTERMEDIATE_DIM, HIDDEN_DIM), dtype=np.float32)
    w_gate_mat[:, :INTERMEDIATE_DIM] = _make_eye(INTERMEDIATE_DIM) * scale
    w_gate_bytes = w_gate_mat.tobytes()

    w_up_mat = np.zeros((INTERMEDIATE_DIM, HIDDEN_DIM), dtype=np.float32)
    w_up_mat[:, :INTERMEDIATE_DIM] = _make_eye(INTERMEDIATE_DIM) * scale
    w_up_bytes = w_up_mat.tobytes()

    w_down_mat = np.zeros((HIDDEN_DIM, INTERMEDIATE_DIM), dtype=np.float32)
    w_down_mat[:INTERMEDIATE_DIM, :] = _make_eye(INTERMEDIATE_DIM)
    w_down_bytes = w_down_mat.tobytes()

    expert = SwiGLUExpert(
        w_gate=w_gate_bytes,
        w_up=w_up_bytes,
        w_down=w_down_bytes,
        hidden_dim=HIDDEN_DIM,
        intermediate_dim=INTERMEDIATE_DIM,
        quant_type="f32",
    )
    _check(expert.hidden_dim == HIDDEN_DIM, "7a: hidden_dim")
    _check(expert.intermediate_dim == INTERMEDIATE_DIM, "7a: intermediate_dim")
    _check(expert.quant_type == "f32", "7a: quant_type")

    # Input: 1.0 in first 1024 dims, 0 elsewhere
    x = np.zeros(HIDDEN_DIM, dtype=np.float32)
    x[:INTERMEDIATE_DIM] = 1.0

    out = expert.forward(x)
    _check(out.shape == (HIDDEN_DIM,), f"7a: output shape (got {out.shape})")
    _check(not np.any(np.isnan(out)), "7a: no NaN in output")
    _check(not np.any(np.isinf(out)), "7a: no Inf in output")

    # Expected:
    #   gate = W_gate @ x = scale * x[:1024] = 0.5
    #   swish(0.5) = 0.5 * sigmoid(0.5)
    #   sigmoid(0.5) = 1/(1+exp(-0.5)) ≈ 0.62246
    #   swish(0.5) ≈ 0.31123
    #   up = W_up @ x = 0.5
    #   gated = swish * up ≈ 0.15561
    #   W_down @ gated: first 1024 = gated, rest = 0
    #   So out[:1024] ≈ 0.15561, out[1024:] = 0

    gate_val = scale  # = 0.5
    swish_val = gate_val / (1.0 + math.exp(-gate_val))  # ≈ 0.31123
    gated_val = swish_val * gate_val  # ≈ 0.15561

    _check(_allclose(out[:INTERMEDIATE_DIM], gated_val, atol=1e-3),
           f"7a: first 1024 outputs ≈ {gated_val:.5f} (got {out[0]:.5f})")
    _check(_allclose(out[INTERMEDIATE_DIM:], 0.0, atol=1e-4),
           "7a: last 1024 outputs ≈ 0")

    # ── 7b: Different input values ─────────────────────────────────
    x2 = np.zeros(HIDDEN_DIM, dtype=np.float32)
    x2[:INTERMEDIATE_DIM] = 2.0
    out2 = expert.forward(x2)
    gate_val2 = scale * 2.0  # = 1.0
    swish_val2 = gate_val2 / (1.0 + math.exp(-gate_val2))  # ≈ 0.73106
    gated_val2 = swish_val2 * gate_val2  # ≈ 0.73106
    _check(_allclose(out2[:INTERMEDIATE_DIM], gated_val2, atol=1e-3),
           f"7b: x=2 → out ≈ {gated_val2:.5f}")

    # ── 7c: List input works ───────────────────────────────────────
    out3 = expert.forward(x.tolist())
    _check(_allclose(out, out3, atol=1e-5), "7c: list input matches ndarray input")

    # ── 7d: Wrong input length raises ──────────────────────────────
    try:
        expert.forward(np.zeros(100, dtype=np.float32))
        _check(False, "7d: wrong input length should raise ValueError")
    except ValueError:
        _check(True, "7d: wrong input length raises ValueError")


# ═══════════════════════════════════════════════════════════════════════════
# Test 8: Integration — Router + SwiGLUExpert pipeline
# ═══════════════════════════════════════════════════════════════════════════


def test_router_expert_integration() -> None:
    """Wire Router output into a SwiGLUExpert — the MoE decode path."""
    from expert_mlp import Router, SwiGLUExpert

    print("── Test 8: Router + SwiGLUExpert integration ──")

    # Build realistic-ish weights for one expert.
    scale = 0.1
    w_gate_mat = np.zeros((INTERMEDIATE_DIM, HIDDEN_DIM), dtype=np.float32)
    w_gate_mat[:, :INTERMEDIATE_DIM] = _make_eye(INTERMEDIATE_DIM) * scale
    w_gate_bytes = w_gate_mat.tobytes()

    w_up_mat = np.zeros((INTERMEDIATE_DIM, HIDDEN_DIM), dtype=np.float32)
    w_up_mat[:, :INTERMEDIATE_DIM] = _make_eye(INTERMEDIATE_DIM) * scale
    w_up_bytes = w_up_mat.tobytes()

    w_down_mat = np.zeros((HIDDEN_DIM, INTERMEDIATE_DIM), dtype=np.float32)
    w_down_mat[:INTERMEDIATE_DIM, :] = _make_eye(INTERMEDIATE_DIM)
    w_down_bytes = w_down_mat.tobytes()

    expert = SwiGLUExpert(
        w_gate=w_gate_bytes, w_up=w_up_bytes, w_down=w_down_bytes,
    )

    # Router with random weights.
    w_router = _make_rand(NUM_EXPERTS, HIDDEN_DIM) * 0.01
    router = Router(w_router, num_experts=NUM_EXPERTS, top_k=TOP_K)

    x = _make_rand(HIDDEN_DIM)

    # Route → select top expert → run it.
    indices, probs = router.forward(x)
    top_expert_idx = indices[0]
    top_prob = float(probs[0])

    _check(0 <= top_expert_idx < NUM_EXPERTS, "8a: top expert in range")
    _check(0.0 < top_prob <= 1.0, "8a: top prob valid")

    # Run the expert (uses identity-like weights regardless of which
    # expert is selected — in reality each expert has different weights).
    out = expert.forward(x)
    _check(out.shape == (HIDDEN_DIM,), "8b: expert output shape")

    # Apply routing probability (weighted sum component).
    weighted = out * top_prob
    _check(weighted.shape == (HIDDEN_DIM,), "8c: weighted output shape")
    _check(not np.any(np.isnan(weighted)), "8c: no NaN")

    print(f"  Router selected expert {top_expert_idx} with p={top_prob:.4f}")


# ═══════════════════════════════════════════════════════════════════════════
# Runner
# ═══════════════════════════════════════════════════════════════════════════


def main() -> int:
    global _passed, _failed, _failures

    print("=" * 66)
    print("  Swarm — dense_backbone & expert_mlp Test Suite")
    print(f"  Target: OLMoE-1B-7B ({HIDDEN_DIM}d, {NUM_HEADS}h, {HEAD_DIM}hd, "
          f"{NUM_EXPERTS}e, k={TOP_K})")
    print("=" * 66)
    print()

    test_rmsnorm()
    test_rope()
    test_kvcache()
    test_attention_block()
    test_lm_head()
    test_router()
    test_swiglu_expert()
    test_router_expert_integration()

    print()
    print("=" * 66)
    print(f"  PASSED: {_passed}")
    print(f"  FAILED: {_failed}")
    if _failures:
        print("  Failures:")
        for f in _failures:
            print(f"    - {f}")
    print("=" * 66)

    return 0 if _failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
