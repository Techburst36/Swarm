#!/usr/bin/env python3
"""
gguf_stream_reader.py — Pure-Python GGUF v2/v3 parser for Swarm Layer 2.

Reads GGUF headers, metadata, and tensor index tables without mmap'ing the
file.  Exposes ``get_tensor_offset(name)`` so Layer 2's storage layer can
issue targeted O_DIRECT reads for any specific expert tensor.

Design rules:
  - Python 3.11+, standard library only (struct, os).
  - Does NOT load tensor data into memory.  File stays open for seek+read
    but this module only reads header/index — tensors are the caller's job.
  - Handles both GGUF v2 and v3.  The only difference relevant here is
    tensor offset width (v2: uint32 padding + uint64 offset; v3: uint64).
  - All GGUF value types supported: u8/i8/u16/i16/u32/i32/f32/bool/string/
    array/u64/i64/f64.

GGUF file layout
----------------
  ┌──────────┬──────────┬───────────────┬───────────────┐
  │ magic    │ version  │ num_tensors   │ num_meta_kv   │
  │  4 B     │ uint32   │   uint64      │   uint64      │
  ├──────────┴──────────┴───────────────┴───────────────┤
  │  Metadata KV pairs (key: string, type: uint32, value) │
  ├──────────────────────────────────────────────────────┤
  │  Tensor info entries (name: string, dims, type, offset)│
  ├──────────────────────────────────────────────────────┤
  │  (optional alignment padding)                        │
  ├──────────────────────────────────────────────────────┤
  │  Tensor data                                         │
  └──────────────────────────────────────────────────────┘
"""

from __future__ import annotations

import os
import struct
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

GGUF_MAGIC = b"GGUF"

# GGUF value type codes → struct format / decoder
_GGUF_TYPE_U8      = 0
_GGUF_TYPE_I8      = 1
_GGUF_TYPE_U16     = 2
_GGUF_TYPE_I16     = 3
_GGUF_TYPE_U32     = 4
_GGUF_TYPE_I32     = 5
_GGUF_TYPE_F32     = 6
_GGUF_TYPE_BOOL    = 7
_GGUF_TYPE_STRING  = 8
_GGUF_TYPE_ARRAY   = 9
_GGUF_TYPE_U64     = 10
_GGUF_TYPE_I64     = 11
_GGUF_TYPE_F64     = 12

_GGUF_VALUE_READERS = {
    _GGUF_TYPE_U8:     lambda f: struct.unpack("<B", f.read(1))[0],
    _GGUF_TYPE_I8:     lambda f: struct.unpack("<b", f.read(1))[0],
    _GGUF_TYPE_U16:    lambda f: struct.unpack("<H", f.read(2))[0],
    _GGUF_TYPE_I16:    lambda f: struct.unpack("<h", f.read(2))[0],
    _GGUF_TYPE_U32:    lambda f: struct.unpack("<I", f.read(4))[0],
    _GGUF_TYPE_I32:    lambda f: struct.unpack("<i", f.read(4))[0],
    _GGUF_TYPE_F32:    lambda f: struct.unpack("<f", f.read(4))[0],
    _GGUF_TYPE_BOOL:   lambda f: struct.unpack("<B", f.read(1))[0] != 0,
    _GGUF_TYPE_U64:    lambda f: struct.unpack("<Q", f.read(8))[0],
    _GGUF_TYPE_I64:    lambda f: struct.unpack("<q", f.read(8))[0],
    _GGUF_TYPE_F64:    lambda f: struct.unpack("<d", f.read(8))[0],
}

# GGML type codes → dtype name and element size
_GGML_TYPES: dict[int, dict[str, Any]] = {
    0:  {"name": "f32",     "elem_size": 4},
    1:  {"name": "f16",     "elem_size": 2},
    2:  {"name": "q4_0",    "elem_size": 0, "block_size": 32,  "block_bytes": 18},
    3:  {"name": "q4_1",    "elem_size": 0, "block_size": 32,  "block_bytes": 20},
    4:  {"name": "q5_0",    "elem_size": 0, "block_size": 32,  "block_bytes": 22},
    5:  {"name": "q5_1",    "elem_size": 0, "block_size": 32,  "block_bytes": 24},
    6:  {"name": "q8_0",    "elem_size": 0, "block_size": 32,  "block_bytes": 34},
    7:  {"name": "q8_1",    "elem_size": 0, "block_size": 32,  "block_bytes": 36},
    8:  {"name": "q2_k",    "elem_size": 0, "block_size": 256, "block_bytes": 82},
    9:  {"name": "q3_k",    "elem_size": 0, "block_size": 256, "block_bytes": 110},
    10: {"name": "q4_k",    "elem_size": 0, "block_size": 256, "block_bytes": 144},
    11: {"name": "q5_k",    "elem_size": 0, "block_size": 256, "block_bytes": 176},
    12: {"name": "q6_k",    "elem_size": 0, "block_size": 256, "block_bytes": 210},
    13: {"name": "q8_k",    "elem_size": 0, "block_size": 256, "block_bytes": 292},
    14: {"name": "iq2_xxs", "elem_size": 0, "block_size": 256, "block_bytes": 66},
    15: {"name": "iq2_xs",  "elem_size": 0, "block_size": 256, "block_bytes": 74},
    16: {"name": "iq3_xxs", "elem_size": 0, "block_size": 256, "block_bytes": 98},
    17: {"name": "iq3_s",   "elem_size": 0, "block_size": 256, "block_bytes": 102},
    18: {"name": "iq2_s",   "elem_size": 0, "block_size": 256, "block_bytes": 86},
    19: {"name": "iq1_s",   "elem_size": 0, "block_size": 256, "block_bytes": 54},
}


@dataclass
class TensorInfo:
    """Metadata for one tensor in a GGUF file."""
    name: str
    dims: list[int]       # reversed from file order (GGUF stores innermost first)
    n_elements: int
    ggml_type: int
    offset: int            # byte offset from file start to tensor data
    size_bytes: int
    dtype_name: str


@dataclass
class GGUFHeader:
    """Parsed GGUF file header."""
    version: int
    num_tensors: int
    num_metadata: int
    metadata: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, TensorInfo] = field(default_factory=dict)
    data_start: int = 0  # byte offset where tensor data begins


class GGUFReader:
    """Read-only GGUF file parser.  Never loads tensor data into memory.

    Parameters
    ----------
    path:
        Path to the GGUF file on disk.
    """

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path).resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"GGUF file not found: {self.path}")

        self.header: GGUFHeader = self._parse_header()

    # ── Public API ──────────────────────────────────────────────────────

    def get_tensor_offset(self, name: str) -> tuple[int, int, str]:
        """Return ``(offset_bytes, length_bytes, dtype)`` for a tensor.

        Parameters
        ----------
        name:
            Tensor name as stored in the GGUF file (e.g.
            ``"blk.0.ffn_gate_experts.0.weight"``).

        Returns
        -------
        tuple[int, int, str]
            ``(offset, size_bytes, dtype_name)`` — offset from file start,
            total bytes of tensor data, and dtype string (e.g. "f32", "q4_k_m").
        """
        info = self.header.tensors.get(name)
        if info is None:
            available = sorted(self.header.tensors.keys())
            raise KeyError(
                f"Tensor '{name}' not found in {self.path.name}. "
                f"Available tensors (first 20): {available[:20]}"
            )
        return info.offset, info.size_bytes, info.dtype_name

    def read_tensor_bytes(self, name: str) -> bytes:
        """Read the full tensor data for *name* into memory.

        For large tensors, prefer using ``get_tensor_offset()`` with
        ``os.pread()`` for O_DIRECT reads.
        """
        info = self.header.tensors[name]
        with open(self.path, "rb") as fh:
            fh.seek(info.offset)
            return fh.read(info.size_bytes)

    def list_tensors(self) -> list[str]:
        """Return sorted list of all tensor names."""
        return sorted(self.header.tensors.keys())

    def tensor_count(self) -> int:
        """Number of tensors in the file."""
        return self.header.num_tensors

    def get_metadata(self, key: str, default: Any = None) -> Any:
        """Return a metadata value by key, or *default* if not present."""
        return self.header.metadata.get(key, default)

    def model_architecture(self) -> str | None:
        """Convenience: return the 'general.architecture' metadata value."""
        return self.header.metadata.get("general.architecture")

    def __repr__(self) -> str:
        return (
            f"GGUFReader({self.path.name!r}, "
            f"v{self.header.version}, "
            f"{self.header.num_tensors} tensors, "
            f"{len(self.header.metadata)} metadata keys)"
        )

    # ── Internal ────────────────────────────────────────────────────────

    def _parse_header(self) -> GGUFHeader:
        with open(self.path, "rb") as f:
            # ── Magic ────────────────────────────────────────────────
            magic = f.read(4)
            if magic != GGUF_MAGIC:
                raise ValueError(
                    f"Not a GGUF file: magic={magic!r}, expected {GGUF_MAGIC!r}"
                )

            # ── Version ──────────────────────────────────────────────
            version = struct.unpack("<I", f.read(4))[0]
            if version not in (2, 3):
                raise ValueError(
                    f"Unsupported GGUF version {version} (expected 2 or 3)"
                )

            # ── Counts ───────────────────────────────────────────────
            num_tensors = struct.unpack("<Q", f.read(8))[0]
            num_metadata = struct.unpack("<Q", f.read(8))[0]

            header = GGUFHeader(
                version=version,
                num_tensors=num_tensors,
                num_metadata=num_metadata,
            )

            # ── Metadata KV pairs ────────────────────────────────────
            for _ in range(num_metadata):
                key = self._read_string(f)
                value_type = struct.unpack("<I", f.read(4))[0]
                value = self._read_value(f, value_type)
                header.metadata[key] = value

            # ── Tensor info entries ──────────────────────────────────
            for _ in range(num_tensors):
                name = self._read_string(f)
                n_dims = struct.unpack("<I", f.read(4))[0]
                dims_raw = list(
                    struct.unpack(f"<{n_dims}Q", f.read(8 * n_dims))
                )
                # GGUF stores dims in reverse (innermost first).  Reverse
                # so dims[0] is the outermost (row) dimension.
                dims = list(reversed(dims_raw))

                n_elements = 1
                for d in dims:
                    n_elements *= d

                ggml_type = struct.unpack("<I", f.read(4))[0]
                offset = struct.unpack("<Q", f.read(8))[0]

                type_info = _GGML_TYPES.get(ggml_type, {})
                dtype_name = type_info.get("name", f"unknown_{ggml_type}")

                # Compute size
                elem_size = type_info.get("elem_size", 0)
                if elem_size > 0:
                    size_bytes = n_elements * elem_size
                else:
                    # Block-quantized: compute from block count
                    block_k = type_info.get("block_size", 256)
                    block_bytes = type_info.get("block_bytes", 0)
                    if block_bytes > 0:
                        blocks = (n_elements + block_k - 1) // block_k
                        size_bytes = blocks * block_bytes
                    else:
                        size_bytes = 0  # unknown

                info = TensorInfo(
                    name=name,
                    dims=dims,
                    n_elements=n_elements,
                    ggml_type=ggml_type,
                    offset=offset,
                    size_bytes=size_bytes,
                    dtype_name=dtype_name,
                )
                header.tensors[name] = info

        return header

    # ── Value readers ──────────────────────────────────────────────────

    def _read_string(self, f) -> str:
        length = struct.unpack("<Q", f.read(8))[0]
        return f.read(length).decode("utf-8", errors="replace")

    def _read_value(self, f, value_type: int):
        if value_type == _GGUF_TYPE_STRING:
            return self._read_string(f)
        elif value_type == _GGUF_TYPE_ARRAY:
            elem_type = struct.unpack('<I', f.read(4))[0]
            length = struct.unpack('<Q', f.read(8))[0]
            if elem_type == _GGUF_TYPE_STRING:
                return [self._read_string(f) for _ in range(length)]
            reader = _GGUF_VALUE_READERS.get(elem_type)
            if reader is None:
                return None
            return [reader(f) for _ in range(length)]
        else:
            reader = _GGUF_VALUE_READERS.get(value_type)
            if reader is None:
                return None
            return reader(f)


# ── GGUF writer (for creating test files) ────────────────────────────────


class GGUFWriter:
    """Write a minimal GGUF v3 file.  For test fixture creation only."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self._metadata: list[tuple[str, int, Any]] = []
        self._tensors: list[tuple[str, list[int], int, bytes]] = []  # name, dims, type, data

    def add_metadata(self, key: str, value: Any) -> None:
        """Add a metadata KV pair.  Only basic types supported for test use."""
        if isinstance(value, str):
            self._metadata.append((key, _GGUF_TYPE_STRING, value))
        elif isinstance(value, bool):
            self._metadata.append((key, _GGUF_TYPE_BOOL, value))
        elif isinstance(value, int) and -2147483648 <= value <= 2147483647:
            self._metadata.append((key, _GGUF_TYPE_U32, value))
        elif isinstance(value, int):
            self._metadata.append((key, _GGUF_TYPE_U64, value))
        elif isinstance(value, float):
            self._metadata.append((key, _GGUF_TYPE_F32, value))
        else:
            raise TypeError(f"Unsupported metadata value type: {type(value)}")

    def add_tensor(self, name: str, dims: list[int],
                   ggml_type: int, data: bytes) -> None:
        """Add a tensor with its raw data bytes."""
        self._tensors.append((name, dims, ggml_type, data))

    def write(self) -> None:
        """Write the GGUF file to disk."""
        with open(self.path, "wb") as f:
            # ── Header ─────────────────────────────────────────────
            f.write(GGUF_MAGIC)
            f.write(struct.pack("<I", 3))  # version 3
            f.write(struct.pack("<Q", len(self._tensors)))
            f.write(struct.pack("<Q", len(self._metadata)))

            # ── Metadata ───────────────────────────────────────────
            for key, vtype, value in self._metadata:
                key_bytes = key.encode("utf-8")
                f.write(struct.pack("<Q", len(key_bytes)))
                f.write(key_bytes)
                f.write(struct.pack("<I", vtype))
                self._write_value(f, vtype, value)

            # ── Tensor infos ────────────────────────────────────────
            # First pass: write tensor info entries, record data offsets
            data_positions: list[int] = []
            current_offset = 0  # will be updated after we know header size

            for name, dims, ggml_type, data in self._tensors:
                name_bytes = name.encode("utf-8")
                f.write(struct.pack("<Q", len(name_bytes)))
                f.write(name_bytes)

                # Dims: store in GGUF reverse order
                rev_dims = list(reversed(dims))
                f.write(struct.pack("<I", len(rev_dims)))
                for d in rev_dims:
                    f.write(struct.pack("<Q", d))

                f.write(struct.pack("<I", ggml_type))
                # Offset placeholder — we'll seek back and fix
                pos = f.tell()
                f.write(struct.pack("<Q", 0))  # placeholder
                data_positions.append(pos)

            # ── Alignment padding (32-byte) ─────────────────────────
            current_pos = f.tell()
            alignment = 32
            pad = (alignment - (current_pos % alignment)) % alignment
            f.write(b"\x00" * pad)

            data_start = f.tell()

            # ── Fix tensor offsets ──────────────────────────────────
            for i, (name, dims, ggml_type, data) in enumerate(self._tensors):
                offset_pos = data_positions[i]
                f.seek(offset_pos)
                f.write(struct.pack("<Q", data_start + sum(
                    len(d) for _, _, _, d in self._tensors[:i]
                )))

            f.seek(0, os.SEEK_END)

            # ── Write tensor data ───────────────────────────────────
            for _, _, _, data in self._tensors:
                f.write(data)

    def _write_value(self, f, vtype: int, value: Any) -> None:
        if vtype == _GGUF_TYPE_STRING:
            b = value.encode("utf-8")
            f.write(struct.pack("<Q", len(b)))
            f.write(b)
        elif vtype == _GGUF_TYPE_BOOL:
            f.write(struct.pack("<B", 1 if value else 0))
        elif vtype == _GGUF_TYPE_U32:
            f.write(struct.pack("<I", value))
        elif vtype == _GGUF_TYPE_U64:
            f.write(struct.pack("<Q", value))
        elif vtype == _GGUF_TYPE_F32:
            f.write(struct.pack("<f", value))
        else:
            raise ValueError(f"Unknown value type: {vtype}")


# ── Self-test ─────────────────────────────────────────────────────────────


def _self_test() -> None:
    import tempfile
    import random
    random.seed(42)

    print("── gguf_stream_reader self-test ──")

    with tempfile.NamedTemporaryFile(suffix=".gguf", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        # Write a minimal GGUF file
        writer = GGUFWriter(tmp_path)
        writer.add_metadata("general.architecture", "swarm-test")
        writer.add_metadata("general.name", "TestModel")
        writer.add_metadata("swarm.num_layers", 4)
        writer.add_metadata("swarm.num_experts", 8)

        # Add F32 tensors: 16×16 matrices = 256 elements = 1024 bytes each
        for layer in range(4):
            for expert in range(8):
                data = struct.pack(
                    f"<256f",
                    *[random.uniform(-1, 1) for _ in range(256)]
                )
                writer.add_tensor(
                    f"blk.{layer}.expert.{expert}.weight",
                    [16, 16],      # dims (rows, cols)
                    0,             # GGML_TYPE_F32
                    data,
                )

        writer.write()

        # Read it back
        reader = GGUFReader(tmp_path)
        print(f"  {reader}")
        print(f"  Architecture: {reader.model_architecture()}")

        # Check tensor lookup
        offset, size, dtype = reader.get_tensor_offset("blk.0.expert.0.weight")
        assert dtype == "f32", f"Expected f32, got {dtype}"
        assert size == 1024, f"Expected 1024, got {size}"
        print(f"  Tensor blk.0.expert.0.weight: offset={offset}, size={size}, dtype={dtype} ✓")

        # Read back and verify
        raw = reader.read_tensor_bytes("blk.0.expert.0.weight")
        assert len(raw) == 1024
        vals = struct.unpack("<256f", raw)
        print(f"  Read back 256 floats: first={vals[0]:.4f}, last={vals[-1]:.4f} ✓")

        # List tensors
        tensors = reader.list_tensors()
        assert len(tensors) == 32, f"Expected 32 tensors, got {len(tensors)}"
        print(f"  {len(tensors)} tensors listed ✓")

        # Metadata
        assert reader.get_metadata("swarm.num_layers") == 4
        print("  Metadata readback ✓")

        print("── All gguf_stream_reader tests passed ──")

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    _self_test()
