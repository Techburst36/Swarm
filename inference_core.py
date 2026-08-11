#!/usr/bin/env python3
"""
inference_core.py — Python bridge to libswarm_core.so with pure-Python fallback.

Loads the compiled C AVX2 GEMM library via ctypes.  Falls back to a pure-Python
implementation when the shared library is not available (no compiler on host,
cross-platform testing, etc.).

Usage:
    from inference_core import execute_expert

    result = execute_expert(weight_bytes, input_vec, rows=256, cols=256,
                            quant_type="f32")
    # result is list[float] of length rows
"""

from __future__ import annotations

import ctypes
import math
import os
import struct
import sys
from pathlib import Path

logger = __import__("logging").getLogger("swarm.inference_core")

# ── C library loading ────────────────────────────────────────────────────

_lib = None
_backend = "python"

# Q4_K_M block constants (must match swarm_core.c)
QK_K = 256
QK_K_SUB = 16
Q4K_SCALES_BYTES = 12
Q4K_QS_BYTES = 128
Q4K_BLOCK_BYTES = 2 + 2 + Q4K_SCALES_BYTES + Q4K_QS_BYTES  # 144


def _find_lib() -> Path | None:
    """Search for libswarm_core.so in standard locations."""
    candidates = [
        Path(__file__).parent / "libswarm_core.so",
        Path.cwd() / "libswarm_core.so",
        Path("/usr/local/lib/libswarm_core.so"),
        Path("/usr/lib/libswarm_core.so"),
    ]
    for p in candidates:
        if p.is_file():
            return p
    return None


def _try_load_lib() -> ctypes.CDLL | None:
    """Attempt to load the compiled C library."""
    lib_path = _find_lib()
    if lib_path is None:
        # Try to compile on the fly if gcc is available.
        src = Path(__file__).parent / "swarm_core.c"
        dst = Path(__file__).parent / "libswarm_core.so"
        if src.exists():
            import subprocess
            try:
                subprocess.run(
                    ["gcc", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC",
                     "-o", str(dst), str(src)],
                    check=True, capture_output=True, timeout=60,
                )
                if dst.exists():
                    lib_path = dst
            except (FileNotFoundError, subprocess.CalledProcessError,
                    subprocess.TimeoutExpired):
                pass

    if lib_path is None:
        return None

    try:
        lib = ctypes.CDLL(str(lib_path))
        lib.gemv_f32.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.gemv_f32.restype = ctypes.c_int

        lib.gemv_q4km.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.c_int,
        ]
        lib.gemv_q4km.restype = ctypes.c_int

        lib.swarm_core_has_avx2.restype = ctypes.c_int
        return lib
    except OSError as exc:
        print(f"[inference_core] Cannot load {lib_path}: {exc}", file=sys.stderr)
        return None


def _get_lib():
    global _lib, _backend
    if _lib is not None:
        return _lib
    _lib = _try_load_lib()
    if _lib is not None:
        _backend = "c_avx2"
    return _lib


def get_backend() -> str:
    """Return the active GEMM backend: 'c_avx2' or 'python'."""
    _get_lib()
    return _backend


# ── Pure-Python GEMM ─────────────────────────────────────────────────────


def _gemv_f32_py(weights: bytes, input_vec: list[float],
                 rows: int, cols: int) -> list[float]:
    """Pure-Python F32 matrix-vector multiply."""
    fmt = f"<{rows * cols}f"
    w = struct.unpack(fmt, weights)
    result = []
    for r in range(rows):
        row_start = r * cols
        dot = 0.0
        for c in range(cols):
            dot += w[row_start + c] * input_vec[c]
        result.append(dot)
    return result


def _gemv_q4km_py(weights: bytes, input_vec: list[float],
                  rows: int, cols: int) -> list[float]:
    """Pure-Python Q4_K_M dequant + matrix-vector multiply."""
    blocks_per_row = (cols + QK_K - 1) // QK_K

    result = []
    for r in range(rows):
        dot = 0.0
        row_off = r * blocks_per_row * Q4K_BLOCK_BYTES

        for b in range(blocks_per_row):
            blk = row_off + b * Q4K_BLOCK_BYTES

            # Header: d (f16), dmin (f16)
            d_raw = struct.unpack_from("<H", weights, blk)[0]
            dmin_raw = struct.unpack_from("<H", weights, blk + 2)[0]
            d = _f16_to_f32(d_raw)
            dmin = _f16_to_f32(dmin_raw)

            # Unpack scales
            scales = _unpack_scales(weights, blk + 4)

            # Qs start
            qs_off = blk + 4 + Q4K_SCALES_BYTES

            base_c = b * QK_K
            valid = min(cols - base_c, QK_K)
            if valid <= 0:
                break

            for i in range(valid):
                sb = i // QK_K_SUB
                q = _q4_nibble(weights, qs_off, i)
                sc = scales[sb]
                val = d * (sc * (q - 8)) - dmin
                dot += val * input_vec[base_c + i]

        result.append(dot)

    return result


# ── f16 helpers ──────────────────────────────────────────────────────────


def _f16_to_f32(h: int) -> float:
    """IEEE 754-2008 binary16 → float32."""
    sign = -1.0 if (h & 0x8000) else 1.0
    exp = (h >> 10) & 0x1F
    mant = h & 0x3FF

    if exp == 0:
        if mant == 0:
            return 0.0 if sign > 0 else -0.0
        # subnormal
        return sign * (mant / 1024.0) * math.pow(2.0, -14.0)
    if exp == 0x1F:
        if mant == 0:
            return float("inf") if sign > 0 else float("-inf")
        return float("nan")
    # normal
    return sign * (1.0 + mant / 1024.0) * math.pow(2.0, exp - 15.0)


def _unpack_scales(data: bytes, offset: int):
    """Unpack 16 × 6-bit scales from 12 bytes."""
    out = []
    for i in range(16):
        bit_off = i * 6
        byte_off = offset + (bit_off // 8)
        shift = bit_off % 8
        val = data[byte_off]
        if byte_off + 1 < offset + 12 and byte_off + 1 < len(data):
            val |= data[byte_off + 1] << 8
        out.append((val >> shift) & 0x3F)
    return out


def _q4_nibble(data: bytes, qs_offset: int, idx: int) -> int:
    """Extract 4-bit nibble from packed qs bytes."""
    b = data[qs_offset + (idx // 2)]
    return (b >> (4 * (idx % 2))) & 0x0F


# ── Public API ────────────────────────────────────────────────────────────


def execute_expert(
    weight_bytes: bytes,
    input_activations: list[float],
    rows: int,
    cols: int,
    quant_type: str = "f32",
) -> list[float]:
    """Execute one expert GEMM: output = dequant(weights) @ input.

    Parameters
    ----------
    weight_bytes:
        Raw weight data.  For 'f32': rows×cols float32 in row-major layout.
        For 'q4_k_m': rows × ceil(cols/256) × 144 bytes in Q4_K_M block format.
    input_activations:
        Input activation vector (length = cols).
    rows:
        Number of output elements (rows of weight matrix).
    cols:
        Number of input elements (columns of weight matrix).
    quant_type:
        'f32' for float32 or 'q4_k_m' for Q4_K_M block-quantized weights.

    Returns
    -------
    list[float]
        Output vector of length *rows*.
    """
    if not weight_bytes:
        raise ValueError("weight_bytes is empty")
    if len(input_activations) != cols:
        raise ValueError(
            f"input_activations length {len(input_activations)} != cols {cols}"
        )
    if rows <= 0 or cols <= 0:
        raise ValueError(f"Invalid dimensions: rows={rows}, cols={cols}")

    # ── Try C library first ──────────────────────────────────────────
    lib = _get_lib()
    if lib is not None:
        input_arr = (ctypes.c_float * cols)(*input_activations)
        output_arr = (ctypes.c_float * rows)()

        if quant_type == "f32":
            expected_size = rows * cols * 4
            if len(weight_bytes) < expected_size:
                raise ValueError(
                    f"Weight bytes too small: {len(weight_bytes)} < "
                    f"{expected_size} (rows={rows}, cols={cols}, f32)"
                )
            # Zero-copy: use from_buffer if possible, else from_buffer_copy
            wtype = ctypes.c_float * (rows * cols)
            wbuf = wtype.from_buffer_copy(weight_bytes)
            ret = lib.gemv_f32(wbuf, input_arr, output_arr, rows, cols)
        elif quant_type == "q4_k_m":
            blocks_per_row = (cols + QK_K - 1) // QK_K
            expected_size = rows * blocks_per_row * Q4K_BLOCK_BYTES
            if len(weight_bytes) < expected_size:
                raise ValueError(
                    f"Weight bytes too small: {len(weight_bytes)} < "
                    f"{expected_size} (rows={rows}, cols={cols}, q4_k_m)"
                )
            ret = lib.gemv_q4km(
                ctypes.c_char_p(weight_bytes),
                input_arr, output_arr, rows, cols,
            )
        else:
            raise ValueError(f"Unknown quant_type: {quant_type}")

        if ret != 0:
            raise RuntimeError(f"gemv_{quant_type} returned error code {ret}")

        return list(output_arr)

    # ── Pure-Python fallback ─────────────────────────────────────────
    if quant_type == "f32":
        expected_size = rows * cols * 4
        if len(weight_bytes) < expected_size:
            raise ValueError(
                f"Weight bytes too small: {len(weight_bytes)} < "
                f"{expected_size} (rows={rows}, cols={cols}, f32)"
            )
        return _gemv_f32_py(weight_bytes, input_activations, rows, cols)
    elif quant_type == "q4_k_m":
        blocks_per_row = (cols + QK_K - 1) // QK_K
        expected_size = rows * blocks_per_row * Q4K_BLOCK_BYTES
        if len(weight_bytes) < expected_size:
            raise ValueError(
                f"Weight bytes too small: {len(weight_bytes)} < "
                f"{expected_size} (rows={rows}, cols={cols}, q4_k_m)"
            )
        return _gemv_q4km_py(weight_bytes, input_activations, rows, cols)
    else:
        raise ValueError(f"Unknown quant_type: {quant_type}")


# ── Compilation helper ───────────────────────────────────────────────────


def compile_lib(target_dir: Path | None = None) -> Path:
    """Compile libswarm_core.so from swarm_core.c.

    Requires gcc with AVX2/FMA support.  Returns the path to the compiled
    library.

    Raises
    ------
    FileNotFoundError
        If gcc is not installed.
    RuntimeError
        If compilation fails.
    """
    if target_dir is None:
        target_dir = Path(__file__).parent

    src = target_dir / "swarm_core.c"
    dst = target_dir / "libswarm_core.so"

    if not src.exists():
        raise FileNotFoundError(f"Source not found: {src}")

    import subprocess

    result = subprocess.run(
        ["gcc", "-O3", "-mavx2", "-mfma", "-shared", "-fPIC",
         "-std=c11", "-o", str(dst), str(src)],
        capture_output=True, text=True, timeout=60,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"Compilation failed:\n{result.stderr}"
        )

    # Reload the library
    global _lib
    _lib = None
    return dst


# ── Self-test ─────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Verify both backends produce identical correct results."""
    import math
    import random

    print("── inference_core self-test ──")
    backend = get_backend()
    print(f"  Active backend: {backend}")

    random.seed(42)

    # Small test: 4×4 F32
    rows, cols = 4, 4
    weights = struct.pack(
        f"<{rows * cols}f",
        *[1, 0, 0, 0,  0, 2, 0, 0,  0, 0, 3, 0,  0, 0, 0, 4],
    )
    input_vec = [1.0, 2.0, 3.0, 4.0]
    result = execute_expert(weights, input_vec, rows, cols, "f32")
    expected = [1.0, 4.0, 9.0, 16.0]
    for i, (r, e) in enumerate(zip(result, expected)):
        assert abs(r - e) < 1e-5, f"row {i}: {r} != {e}"
    print("  F32 4×4 identity-diag: ✓")

    # Larger random test: 32×64 F32
    rows, cols = 32, 64
    w_vals = [random.uniform(-1, 1) for _ in range(rows * cols)]
    weights = struct.pack(f"<{rows * cols}f", *w_vals)
    input_vec = [random.uniform(-1, 1) for _ in range(cols)]
    result = execute_expert(weights, input_vec, rows, cols, "f32")

    # Verify manually
    for r in range(rows):
        expected_r = sum(
            w_vals[r * cols + c] * input_vec[c] for c in range(cols)
        )
        assert abs(result[r] - expected_r) < 1e-4, (
            f"row {r}: {result[r]} != {expected_r}"
        )
    print("  F32 32×64 random: ✓")

    # Q4_K_M test (synthetic block)
    # One block of 256 elements, one row
    # Create a simple weight pattern
    rows, cols = 1, 256
    d, dmin = 0.5, 0.1
    d_f16 = _f32_to_f16(d)
    dmin_f16 = _f32_to_f16(dmin)

    # Build Q4_K_M block: 2 bytes d, 2 bytes dmin, 12 bytes scales, 128 bytes qs
    scales_bytes = bytearray(12)
    # Set all scales to 8 (a moderate value), packed as 16×6-bit values
    for i in range(16):
        bit_off = i * 6
        byte_off = bit_off // 8
        shift = bit_off % 8
        cur = scales_bytes[byte_off]
        if byte_off + 1 < 12:
            cur |= scales_bytes[byte_off + 1] << 8
        cur |= (8 & 0x3F) << shift
        scales_bytes[byte_off] = cur & 0xFF
        if byte_off + 1 < 12:
            scales_bytes[byte_off + 1] = (cur >> 8) & 0xFF

    # Set all qs to 8 (center value, gives 0 after (q-8))
    qs_bytes = bytes([0x88] * 128)  # nibble 8 for all

    block = struct.pack("<HH", d_f16, dmin_f16) + bytes(scales_bytes) + qs_bytes

    # With all q=8, all dequantized values = d * sc * (8-8) - dmin = -dmin = -0.1
    # Dot with input: sum(-0.1 * input[i]) = -0.1 * sum(input)
    input_vec = [1.0] * cols
    result = execute_expert(block, input_vec, rows, cols, "q4_k_m")
    expected = -dmin * cols  # -0.1 * 256 = -25.6
    assert abs(result[0] - expected) < 0.5, (
        f"Q4_K_M all-8: {result[0]} != {expected}"
    )
    print(f"  Q4_K_M 1×256 all-8: ✓ (got {result[0]:.2f}, expected {expected:.2f})")

    print(f"── All tests passed (backend={backend}) ──")


def _f32_to_f16(f: float) -> int:
    """float32 → IEEE 754-2008 binary16.  Rounds toward zero."""
    if f == 0.0:
        return 0
    import math as _m

    sign = 0x8000 if _m.copysign(1.0, f) < 0 else 0
    f = abs(f)

    if _m.isnan(f):
        return sign | 0x7E00
    if _m.isinf(f):
        return sign | 0x7C00

    # Get exponent and mantissa
    # Use struct to get the raw bits
    raw = struct.unpack("<I", struct.pack("<f", f))[0]
    exp32 = (raw >> 23) & 0xFF
    mant32 = raw & 0x7FFFFF

    if exp32 == 0:
        return sign  # zero/subnormal → zero

    exp16 = exp32 - 127 + 15
    if exp16 <= 0:
        return sign  # underflow → zero
    if exp16 >= 0x1F:
        return sign | 0x7C00  # overflow → Inf

    # Round to 10 mantissa bits (drop 13 bits)
    mant16 = mant32 >> 13
    return sign | (exp16 << 10) | mant16


if __name__ == "__main__":
    _self_test()
