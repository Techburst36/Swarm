#!/usr/bin/env python3
"""
test_full_swarm_local.py — Full 5-layer Swarm integration test.

Boots a 3-node simulated cluster on localhost and exercises the entire stack:
  Layer 1 (inference_core)   — AVX2 GEMM or Python fallback
  Layer 2 (storage + cache) — GGUFExpertStore + ExpertCache
  Layer 3 (node_identity)   — FleetTable discovery
  Layer 4 (rpc/shard/gang/pipeline/failover) — distributed scheduler
  Layer 5 (api_server)      — OpenAI-compatible HTTP endpoint

Usage:
    python3 test_full_swarm_local.py          # run all tests
    python3 test_full_swarm_local.py --verbose  # debug logging
    python3 test_full_swarm_local.py --keep-file # don't delete test GGUF
"""

from __future__ import annotations

import asyncio

# Python 3.10 compatibility shim for asyncio.timeout
if not hasattr(asyncio, "timeout"):
    import contextlib
    class _TimeoutCompat:
        def __init__(self, delay):
            self.delay = delay
        async def __aenter__(self):
            self._task = asyncio.current_task()
            self._handle = asyncio.get_running_loop().call_later(self.delay, self._task.cancel)
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            self._handle.cancel()
            if exc_type is asyncio.CancelledError:
                raise asyncio.TimeoutError()
            return False
    asyncio.timeout = _TimeoutCompat

import contextlib
import json
import logging
import os
import struct
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# ── Helpers ────────────────────────────────────────────────────────────────


def _contextlib_suppress(*exc):
    return contextlib.suppress(*exc)


# ═══════════════════════════════════════════════════════════════════════════
# Test GGUF file creation
# ═══════════════════════════════════════════════════════════════════════════


def create_test_gguf(path: str, num_layers: int = 4, num_experts: int = 8,
                     expert_rows: int = 16, expert_cols: int = 16) -> str:
    """Create a minimal GGUF v3 file with fake expert weight tensors.

    Each expert tensor is a F32 matrix of shape [expert_rows, expert_cols].
    Total file size ≈ num_layers * num_experts * expert_rows * expert_cols * 4
    plus GGUF header overhead.

    Returns the path to the created file.
    """
    from gguf_stream_reader import GGUFWriter

    writer = GGUFWriter(path)

    # Metadata
    writer.add_metadata("general.architecture", "swarm-test")
    writer.add_metadata("general.name", "SwarmTestModel")
    writer.add_metadata("swarm.num_layers", num_layers)
    writer.add_metadata("swarm.num_experts", num_experts)

    import random
    random.seed(12345)

    # Add expert tensors: each is a small F32 matrix.
    # Weights are scaled to keep activations bounded across layers.
    for layer in range(num_layers):
        for expert in range(num_experts):
            n_elements = expert_rows * expert_cols
            # Identity-like matrix scaled by 0.5: output stays bounded
            vals = []
            for r in range(expert_rows):
                for c in range(expert_cols):
                    if r == c:
                        vals.append(0.5 / expert_cols)
                    else:
                        vals.append(0.0)
            data = struct.pack(f"<{n_elements}f", *vals)
            writer.add_tensor(
                f"blk.{layer}.expert.{expert}.weight",
                [expert_rows, expert_cols],
                0,  # GGML_TYPE_F32
                data,
            )

    writer.write()
    file_size = os.path.getsize(path)
    print(f"  Created test GGUF: {path}")
    print(f"    {num_layers} layers × {num_experts} experts × "
          f"{expert_rows}×{expert_cols} F32 = {file_size} bytes")
    return path


# ═══════════════════════════════════════════════════════════════════════════
# Compute stage factory
# ═══════════════════════════════════════════════════════════════════════════


def make_compute_stage(store: Any, expert_rows: int = 16,
                       expert_cols: int = 16) -> Any:
    """Return a compute_stage callback that reads expert weights from
    *store* and runs GEMM via inference_core.execute_expert.

    The callback signature matches PipelineCoordinator's expectation:
        (activation_bytes, layer_start, layer_end) -> bytes
    """
    from inference_core import execute_expert

    async def _compute_stage(
        activation_bytes: bytes,
        layer_start: int,
        layer_end: int,
    ) -> bytes:
        # Decode input: list of floats
        n_floats = len(activation_bytes) // 4
        input_vec = list(struct.unpack(f"<{n_floats}f", activation_bytes))

        # Process each layer
        result = input_vec
        for layer in range(layer_start, layer_end):
            # For each expert in this layer, read weights and compute
            # A real MoE layer would gate and combine experts; here we
            # just read one expert and multiply as proof of pipeline flow.
            expert = 0  # always read expert 0 for the test
            weight_bytes = await store.read_expert(layer, expert)

            # Execute the GEMM: [expert_rows, expert_cols] @ [expert_cols] → [expert_rows]
            out_vec = execute_expert(
                weight_bytes, result, expert_rows, expert_cols, "f32"
            )
            result = out_vec

        # Encode output
        return struct.pack(f"<{len(result)}f", *result)

    return _compute_stage


# ═══════════════════════════════════════════════════════════════════════════
# Test 1: inference_core unit tests
# ═══════════════════════════════════════════════════════════════════════════


def test_inference_core() -> None:
    """Verify the inference_core functions (C or Python backend)."""
    from inference_core import (
        execute_expert, get_backend, _gemv_f32_py, _gemv_q4km_py,
        _f32_to_f16, _f16_to_f32,
    )
    import math
    import random

    print("── Test 1: inference_core ──")
    backend = get_backend()
    print(f"  Backend: {backend}")

    random.seed(42)

    # ── F32 GEMM small identity ────────────────────────────────────
    rows, cols = 4, 4
    weights = struct.pack(
        f"<{rows * cols}f",
        1, 0, 0, 0,
        0, 2, 0, 0,
        0, 0, 3, 0,
        0, 0, 0, 4,
    )
    input_vec = [1.0, 2.0, 3.0, 4.0]
    result = execute_expert(weights, input_vec, rows, cols, "f32")
    expected = [1.0, 4.0, 9.0, 16.0]
    for i, (r, e) in enumerate(zip(result, expected)):
        assert abs(r - e) < 1e-5, f"row {i}: {r} != {e}"
    print("  F32 4×4 identity-diag: ✓")

    # ── F32 GEMM random ───────────────────────────────────────────
    rows, cols = 16, 16
    w_vals = [random.uniform(-1, 1) for _ in range(rows * cols)]
    weights = struct.pack(f"<{rows * cols}f", *w_vals)
    input_vec = [random.uniform(-1, 1) for _ in range(cols)]
    result = execute_expert(weights, input_vec, rows, cols, "f32")
    for r in range(rows):
        expected_r = sum(w_vals[r * cols + c] * input_vec[c] for c in range(cols))
        assert abs(result[r] - expected_r) < 1e-4
    print("  F32 16×16 random: ✓")

    # ── Q4_K_M synthetic ──────────────────────────────────────────
    # Single block of 256 elements, all nibbles = 8 (center value)
    # Dequant: val = d * sc * (8-8) - dmin = -dmin
    rows, cols = 1, 256
    d, dmin = 0.5, 0.1
    d_f16 = _f32_to_f16(d)
    dmin_f16 = _f32_to_f16(dmin)

    # Build scales: all 8
    scales_bytes = bytearray(12)
    for i in range(16):
        bit_off = i * 6
        byte_off = bit_off >> 3
        shift = bit_off & 7
        cur = scales_bytes[byte_off]
        if byte_off + 1 < 12:
            cur |= scales_bytes[byte_off + 1] << 8
        cur |= (8 & 0x3F) << shift
        scales_bytes[byte_off] = cur & 0xFF
        if byte_off + 1 < 12:
            scales_bytes[byte_off + 1] = (cur >> 8) & 0xFF

    qs = bytes([0x88] * 128)  # all nibbles = 8
    block = struct.pack("<HH", d_f16, dmin_f16) + bytes(scales_bytes) + qs

    input_vec = [1.0] * cols
    result = execute_expert(block, input_vec, rows, cols, "q4_k_m")
    expected = -dmin * cols  # -0.1 * 256 = -25.6
    assert abs(result[0] - expected) < 0.5, f"Q4_K_M: {result[0]} != {expected}"
    print(f"  Q4_K_M 1×256 all-8: ✓ (got {result[0]:.2f}, expected {expected:.2f})")

    # ── f16 conversion round-trip ──────────────────────────────────
    for val in [0.0, 1.0, -1.0, 0.5, -0.5, 3.14159, 65504.0]:
        rt = _f16_to_f32(_f32_to_f16(val))
        # f16 has ~3.3 decimal digits of precision
        rel_err = abs(rt - val) / max(abs(val), 1e-8) if abs(val) > 1e-8 else abs(rt)
        assert rel_err < 0.001, f"f16 round-trip: {val} → {rt} (err={rel_err})"
    print("  f16 round-trip: ✓")

    print("── Test 1 passed ──\n")


# ═══════════════════════════════════════════════════════════════════════════
# Test 2: gguf_stream_reader
# ═══════════════════════════════════════════════════════════════════════════


def test_gguf_reader(gguf_path: str, num_layers: int, num_experts: int) -> None:
    """Verify the GGUF reader can parse the test file."""
    from gguf_stream_reader import GGUFReader

    print("── Test 2: gguf_stream_reader ──")
    reader = GGUFReader(gguf_path)
    print(f"  {reader}")

    assert reader.header.version == 3
    assert reader.header.num_tensors == num_layers * num_experts
    assert reader.get_metadata("general.architecture") == "swarm-test"

    # Check tensor lookup
    offset, size, dtype = reader.get_tensor_offset("blk.0.expert.0.weight")
    assert dtype == "f32", f"Expected f32, got {dtype}"
    assert size == 16 * 16 * 4  # 1024 bytes
    print(f"  blk.0.expert.0.weight: offset={offset}, size={size}, dtype={dtype} ✓")

    # Read and verify data
    raw = reader.read_tensor_bytes("blk.0.expert.0.weight")
    assert len(raw) == 1024
    vals = struct.unpack("<256f", raw)
    # Diagonal element: 0.5 / 16 = 0.03125
    assert abs(vals[0] - 0.03125) < 1e-5, f"Unexpected first val: {vals[0]}"
    # Off-diagonal (index 1 = row 0, col 1): 0.0
    assert abs(vals[1] - 0.0) < 1e-6, f"Unexpected val[1]: {vals[1]}"
    print(f"  Data integrity: vals[0]={vals[0]:.5f}, vals[1]={vals[1]:.5f} ✓")

    # Check all tensors exist
    for layer in range(num_layers):
        for expert in range(num_experts):
            name = f"blk.{layer}.expert.{expert}.weight"
            assert name in reader.header.tensors, f"Missing: {name}"
    print(f"  All {num_layers * num_experts} tensors accessible ✓")

    print("── Test 2 passed ──\n")


# ═══════════════════════════════════════════════════════════════════════════
# Test 3: GGUFExpertStore
# ═══════════════════════════════════════════════════════════════════════════


async def test_expert_store(gguf_path: str, num_layers: int,
                            num_experts: int) -> None:
    """Verify the GGUFExpertStore reads correct data."""
    from simulate_fleet import GGUFExpertStore

    print("── Test 3: GGUFExpertStore ──")
    store = GGUFExpertStore(gguf_path)

    # Read expert (0, 0)
    data_expert_00 = await store.read_expert(0, 0)
    assert len(data_expert_00) == 1024, f"Expected 1024 bytes, got {len(data_expert_00)}"
    vals = struct.unpack("<256f", data_expert_00)
    # Diagonal element: 0.5 / 16 = 0.03125
    assert abs(vals[0] - 0.03125) < 1e-5
    # Off-diagonal (index 1): 0.0
    assert abs(vals[1] - 0.0) < 1e-6
    print(f"  Expert (0,0): {len(data_expert_00)} bytes, vals[0]={vals[0]:.5f}, vals[1]={vals[1]:.5f} ✓")

    # Read expert (7, 7) — last one
    data_expert_last = await store.read_expert(num_layers - 1, num_experts - 1)
    assert len(data_expert_last) == 1024
    vals = struct.unpack("<256f", data_expert_last)
    # All layers/experts have the same scaled-identity weights
    assert abs(vals[0] - 0.03125) < 1e-5
    print(f"  Expert ({num_layers-1},{num_experts-1}): vals[0]={vals[0]:.5f} ✓")

    # Read same expert again (cached offset)
    data2 = await store.read_expert(0, 0)
    assert data2 == data_expert_00, "Repeated read returned different data"
    print("  Repeated read consistent ✓")

    await store.close()
    print("── Test 3 passed ──\n")


# ═══════════════════════════════════════════════════════════════════════════
# Test 4: Layer 1 → Layer 2 pipeline (compute_stage integration)
# ═══════════════════════════════════════════════════════════════════════════


async def test_layer1_layer2(gguf_path: str) -> None:
    """Verify that compute_stage reads from GGUF and runs GEMM correctly."""
    from simulate_fleet import GGUFExpertStore

    print("── Test 4: Layer 1 + Layer 2 integration ──")
    store = GGUFExpertStore(gguf_path)

    expert_rows, expert_cols = 16, 16
    compute_fn = make_compute_stage(store, expert_rows, expert_cols)

    # Create input activation: 16 floats, 4 bytes each = 64 bytes
    input_vec = [float(i + 1) for i in range(expert_cols)]
    activation_bytes = struct.pack(f"<{expert_cols}f", *input_vec)

    # Run one layer (layer 0 only)
    result_bytes = await compute_fn(activation_bytes, 0, 1)
    result_vec = list(struct.unpack(f"<{expert_rows}f", result_bytes))
    assert len(result_vec) == expert_rows
    # With scaled-identity weights (0.5/16 diag, 0 off-diag):
    # output[r] = (0.5/16) * input[r] = 0.03125 * input[r]
    for r in range(expert_rows):
        expected = 0.5 / expert_cols * input_vec[r]
        assert abs(result_vec[r] - expected) < 1e-5, (
            f"Row {r}: {result_vec[r]:.6f} != {expected:.6f}"
        )
    print(f"  Layer 0: input=[{input_vec[0]:.1f}, …] → output=[{result_vec[0]:.5f}, …] ✓")
    print("  Layer 0 GEMM verified manually ✓")

    await store.close()
    print("── Test 4 passed ──\n")


# ═══════════════════════════════════════════════════════════════════════════
# Test 5: Full 5-layer fleet
# ═══════════════════════════════════════════════════════════════════════════


async def test_full_fleet(gguf_path: str, num_layers: int,
                          num_experts: int) -> None:
    """Boot a 3-node fleet with real GGUF store on node 0 and simulated
    stores on nodes 1-2, then verify the pipeline runs end-to-end."""
    from simulate_fleet import SimulatedFleet, GGUFExpertStore
    from storage_io import SimulatedExpertStore

    print("── Test 5: Full 5-layer fleet ──")

    expert_rows, expert_cols = 16, 16
    expert_size = expert_rows * expert_cols * 4  # 1024 bytes

    # Create a flat expert weights file for SimulatedExpertStore
    # so all nodes see the same weights as the GGUF.
    import tempfile as _tf
    flat_path = os.path.join(_tf.gettempdir(), f"swarm_flat_{os.getpid()}.bin")
    with open(flat_path, "wb") as f:
        for layer in range(num_layers):
            for expert in range(num_experts):
                buf = bytearray(expert_size)
                for r in range(expert_rows):
                    for c in range(expert_cols):
                        idx = r * expert_cols + c
                        if r == c:
                            struct.pack_into("<f", buf, idx * 4, 0.5 / expert_cols)
                        else:
                            struct.pack_into("<f", buf, idx * 4, 0.0)
                f.write(buf)

    # Build store factories: node 0 = real GGUF, nodes 1-2 = file-backed simulated
    def _factory_0(idx: int):
        return GGUFExpertStore(gguf_path)

    def _factory_sim(idx: int):
        bw = 280  # eMMC-class for simulated peers
        return SimulatedExpertStore(
            expert_size_bytes=expert_size,
            num_layers=num_layers,
            num_experts=num_experts,
            bandwidth_mbps=float(bw),
            latency_ms=0.5,
            seed=idx * 1000,
            backing_file=flat_path,
        )

    factories = [_factory_0, _factory_sim, _factory_sim]

    fleet = SimulatedFleet(
        num_nodes=3,
        base_port=24800,
        expert_store_factories=factories,
        num_experts=num_experts,
        num_layers=num_layers,
        settle_window=0.3,
        own_storage_bandwidths=[3200, 280, 280],
    )

    await fleet.start()

    # Wire up compute_stage for each pipeline node AFTER start
    for i in range(3):
        store = fleet.get_store(i)
        cb = make_compute_stage(store, expert_rows, expert_cols)
        fleet.get_pipeline(i)._compute_stage = cb

    # Verify convergence
    assignment = fleet.get_assignment(0)
    assert assignment is not None, "Fleet did not converge"
    assert len(assignment.node_counts) == 3
    print(f"  Fleet converged: {len(assignment.node_counts)} nodes")
    print(f"  Fleet hash: {assignment.fleet_hash[:16]}…")
    print(f"  Assignment:\n{assignment.summary()}")

    # Run a pipeline request through node 0
    pipe = fleet.get_pipeline(0)

    input_vec = [float(i + 1) for i in range(expert_cols)]
    activation_bytes = struct.pack(f"<{expert_cols}f", *input_vec)

    try:
        result_bytes = await pipe.run_pipeline(
            assignment=assignment,
            request_id=1,
            initial_activation=activation_bytes,
            timeout=10.0,
        )
        result_vec = list(struct.unpack(f"<{expert_rows}f", result_bytes))
        # After 8 layers of scaled-identity: output = (0.5/16)^8 * input
        scale = (0.5 / expert_cols) ** num_layers
        for r in range(expert_rows):
            expected = scale * input_vec[r]
            assert abs(result_vec[r] - expected) < 1e-5, (
                f"Row {r}: {result_vec[r]:.10f} != {expected:.10f}"
            )
        print(f"  Pipeline result: {[round(v, 10) for v in result_vec[:4]]}… ✓")
        print("  Pipeline completed successfully ✓")
    except Exception as exc:
        print(f"  Pipeline failed: {exc}")
        raise

    await fleet.stop()
    with _contextlib_suppress(OSError):
        os.unlink(flat_path)
    print("── Test 5 passed ──\n")


# ═══════════════════════════════════════════════════════════════════════════
# Test 6: API server integration
# ═══════════════════════════════════════════════════════════════════════════


async def test_api_server(gguf_path: str, num_layers: int,
                          num_experts: int) -> None:
    """Start the fleet + API server, issue an HTTP request, verify response."""
    from simulate_fleet import SimulatedFleet, GGUFExpertStore
    from storage_io import SimulatedExpertStore
    from node_identity import FleetTable
    from failover import FailoverCoordinator
    from api_server import ApiServer, InstanceManager, stub_generate

    print("── Test 6: API server integration ──")

    expert_rows, expert_cols = 16, 16
    expert_size = expert_rows * expert_cols * 4

    # Create a flat expert weights file for SimulatedExpertStore
    import tempfile as _tf2
    flat_path = os.path.join(_tf2.gettempdir(), f"swarm_flat_api_{os.getpid()}.bin")
    with open(flat_path, "wb") as f:
        for layer in range(num_layers):
            for expert in range(num_experts):
                buf = bytearray(expert_size)
                for r in range(expert_rows):
                    for c in range(expert_cols):
                        idx = r * expert_cols + c
                        if r == c:
                            struct.pack_into("<f", buf, idx * 4, 0.5 / expert_cols)
                        else:
                            struct.pack_into("<f", buf, idx * 4, 0.0)
                f.write(buf)

    def _factory_0(idx: int):
        return GGUFExpertStore(gguf_path)

    def _factory_sim(idx: int):
        return SimulatedExpertStore(
            expert_size_bytes=expert_size,
            num_layers=num_layers,
            num_experts=num_experts,
            bandwidth_mbps=280.0,
            latency_ms=0.5,
            seed=idx * 1000,
            backing_file=flat_path,
        )

    factories = [_factory_0, _factory_sim, _factory_sim]

    fleet = SimulatedFleet(
        num_nodes=3,
        base_port=24900,
        expert_store_factories=factories,
        num_experts=num_experts,
        num_layers=num_layers,
        settle_window=0.3,
        own_storage_bandwidths=[3200, 280, 280],
    )

    await fleet.start()

    # Wire compute_stage for each pipeline node AFTER start
    for i in range(3):
        store = fleet.get_store(i)
        cb = make_compute_stage(store, expert_rows, expert_cols)
        fleet.get_pipeline(i)._compute_stage = cb

    # Use node 0's failover for the API server
    failover = fleet.get_failover(0)
    fleet_table = fleet._fleet_tables[0]

    # Create a new failover on fleet_table directly (the one in fleet
    # already consumed its join events, but we need the same fleet_table
    # for InstanceManager to work with)
    # Actually, the failover in fleet already has the assignment.
    # Let's use it directly.

    instance = InstanceManager(
        failover=failover,
        model_name="swarm-test-model",
        generate_fn=stub_generate,
        reshard_grace=0.1,
    )
    await instance.start()

    server = ApiServer(
        instance=instance,
        bind="127.0.0.1",
        port=8000,
        max_concurrent=16,
    )
    await server.start()

    # Wait for fleet to be ready
    await asyncio.sleep(0.5)
    assert instance.ready, "Instance not ready"
    print(f"  API server listening on 127.0.0.1:8000 (model={instance.model_name}) ✓")

    # Issue HTTP request via asyncio
    import asyncio as _asyncio

    # Build HTTP request manually (no httpx/requests dependency)
    body = json.dumps({
        "model": "swarm-test-model",
        "messages": [
            {"role": "user", "content": "Hello, Swarm! Test the full pipeline."}
        ],
        "max_tokens": 64,
        "temperature": 0.7,
    })

    request_bytes = (
        f"POST /v1/chat/completions HTTP/1.1\r\n"
        f"Host: 127.0.0.1:8000\r\n"
        f"Content-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Connection: close\r\n"
        f"\r\n"
        f"{body}"
    ).encode("utf-8")

    reader, writer = await _asyncio.open_connection("127.0.0.1", 8000)
    writer.write(request_bytes)
    await writer.drain()

    # Read response
    response = b""
    try:
        async with _asyncio.timeout(10):
            while True:
                chunk = await reader.read(65536)
                if not chunk:
                    break
                response += chunk
    except _asyncio.TimeoutError:
        pass

    writer.close()
    await writer.wait_closed()

    response_str = response.decode("utf-8", errors="replace")
    print(f"  HTTP response status: {response_str[:100]}...")

    # Parse response
    header_end = response_str.find("\r\n\r\n")
    assert header_end > 0, f"No HTTP header found in response: {response_str[:200]}"
    headers_part = response_str[:header_end]
    body_part = response_str[header_end + 4:]

    # Check status
    assert "200 OK" in headers_part, f"Expected 200 OK, got: {headers_part.split(chr(13))[0]}"

    # Parse JSON body
    resp_json = json.loads(body_part)
    assert "choices" in resp_json, f"No choices in response: {resp_json}"
    content = resp_json["choices"][0]["message"]["content"]
    print(f"  Response content: {content[:100]}...")
    assert "[STUB" in content, f"Expected STUB prefix, got: {content[:50]}"
    assert "Fleet:" in content or "fleet" in content.lower()

    print("  API returned valid chat completion ✓")

    # Test /health endpoint
    reader2, writer2 = await _asyncio.open_connection("127.0.0.1", 8000)
    writer2.write(b"GET /health HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nConnection: close\r\n\r\n")
    await writer2.drain()
    health_resp = b""
    try:
        async with _asyncio.timeout(5):
            while True:
                chunk = await reader2.read(65536)
                if not chunk:
                    break
                health_resp += chunk
    except _asyncio.TimeoutError:
        pass
    writer2.close()
    await writer2.wait_closed()

    health_str = health_resp.decode("utf-8", errors="replace")
    hh = health_str.find("\r\n\r\n")
    health_body = json.loads(health_str[hh + 4:]) if hh > 0 else {}
    assert health_body.get("status") in ("ok", "degraded")
    assert health_body.get("fleet_size", 0) > 0
    print(f"  Health check: status={health_body.get('status')}, fleet_size={health_body.get('fleet_size')} ✓")

    # Test /v1/models
    reader3, writer3 = await _asyncio.open_connection("127.0.0.1", 8000)
    writer3.write(b"GET /v1/models HTTP/1.1\r\nHost: 127.0.0.1:8000\r\nConnection: close\r\n\r\n")
    await writer3.drain()
    models_resp = b""
    try:
        async with _asyncio.timeout(5):
            while True:
                chunk = await reader3.read(65536)
                if not chunk:
                    break
                models_resp += chunk
    except _asyncio.TimeoutError:
        pass
    writer3.close()
    await writer3.wait_closed()

    models_str = models_resp.decode("utf-8", errors="replace")
    mh = models_str.find("\r\n\r\n")
    models_body = json.loads(models_str[mh + 4:]) if mh > 0 else {}
    assert models_body.get("data", [{}])[0].get("id") == "swarm-test-model"
    print(f"  Models list: {models_body['data'][0]['id']} ✓")

    # Shutdown
    await server.stop()
    await instance.stop()
    await fleet.stop()

    print("── Test 6 passed ──\n")


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════


import argparse as _argparse


async def amain(args: _argparse.Namespace) -> int:
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    # Quiet the fleet chatter unless verbose
    if not args.verbose:
        for name in [
            "swarm.rpc", "swarm.node_identity", "swarm.gang_sync",
            "swarm.pipeline", "swarm.failover", "swarm.api_server",
            "swarm.simulate_fleet", "swarm.storage_io", "swarm.expert_cache",
        ]:
            logging.getLogger(name).setLevel(logging.WARNING)

    print("=" * 72)
    print("  Swarm — Full 5-Layer Integration Test")
    print("=" * 72)
    print()

    num_layers = 8   # GGUF layers = fleet num_experts (pipeline reuses shard assignment)
    num_experts = 8

    # Compile C library if possible
    from inference_core import compile_lib as _cl, get_backend as _gb
    try:
        lib_path = _cl()
        print(f"  Compiled: {lib_path}")
    except (FileNotFoundError, RuntimeError) as exc:
        print(f"  Note: C library not compiled ({exc}) — using Python fallback")

    backend = _gb()
    print(f"  GEMM backend: {backend}")

    # Create test GGUF file
    gguf_path = args.gguf_path or os.path.join(
        tempfile.gettempdir(), f"swarm_test_{os.getpid()}.gguf"
    )
    create_test_gguf(gguf_path, num_layers, num_experts)

    try:
        # ── Run tests ──────────────────────────────────────────────
        test_inference_core()
        test_gguf_reader(gguf_path, num_layers, num_experts)
        await test_expert_store(gguf_path, num_layers, num_experts)
        await test_layer1_layer2(gguf_path)
        await test_full_fleet(gguf_path, num_layers, num_experts)
        await test_api_server(gguf_path, num_layers, num_experts)

        print("=" * 72)
        print("  ALL TESTS PASSED")
        print("=" * 72)
        return 0

    finally:
        if not args.keep_file:
            with _contextlib_suppress(OSError):
                os.unlink(gguf_path)
                print(f"\n  Cleaned up: {gguf_path}")


def main() -> int:
    parser = _argparse.ArgumentParser(
        description="Swarm full 5-layer integration test"
    )
    parser.add_argument(
        "--verbose", action="store_true",
        help="Enable DEBUG logging for all modules",
    )
    parser.add_argument(
        "--keep-file", action="store_true",
        help="Don't delete the test GGUF file after the test",
    )
    parser.add_argument(
        "--gguf-path", type=str, default=None,
        help="Path for the test GGUF file (default: temp directory)",
    )
    args = parser.parse_args()
    return asyncio.run(amain(args))


if __name__ == "__main__":
    sys.exit(main())
