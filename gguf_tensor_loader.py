#!/usr/bin/env python3
"""
gguf_tensor_loader.py — GGUF weight loader and tensor mapper for Swarm.

Reads OLMoE-1B-7B tensors from a GGUF file via ``gguf_stream_reader`` and
maps them to the layer objects defined in ``dense_backbone.py`` and
``expert_mlp.py``.

Supports both real GGUF files and test fixture files with reduced dimensions.

Design rules:
  - Python 3.10+, standard library + numpy + existing project modules.
  - Loads dense parameters into RAM as numpy arrays or layer instances.
  - Expert weights stay in the GGUF file — only offsets are cached.  The
    ExpertStore reads them on demand via ``get_expert_offset()``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

import numpy as np

from dense_backbone import RMSNorm, AttentionBlock, LMHead, KVCache, RoPE
from expert_mlp import Router, SwiGLUExpert
from gguf_stream_reader import GGUFReader


@dataclass
class ModelConfig:
    """Dimensions and metadata for an OLMoE-style MoE model."""
    num_layers: int = 28
    hidden_dim: int = 2048
    intermediate_dim: int = 1024
    num_experts: int = 64
    top_k: int = 8
    num_heads: int = 16
    num_kv_heads: int = 16
    head_dim: int = 128
    vocab_size: int = 102848
    max_seq_len: int = 2048
    rotary_theta: float = 10000.0
    eps: float = 1e-5
    tied_embeddings: bool = False  # OLMoE ties token_embd / output


class GGUFTensorLoader:
    """Loads and maps GGUF tensors to Swarm layer objects.

    Parameters
    ----------
    gguf_path:
        Path to the GGUF file on disk.
    config:
        Model dimensions.  If ``None``, auto-detects from GGUF metadata
        and tensor shapes.
    """

    def __init__(
        self,
        gguf_path: str,
        config: ModelConfig | None = None,
    ) -> None:
        if not os.path.isfile(gguf_path):
            raise FileNotFoundError(f"GGUF file not found: {gguf_path}")

        self._path = gguf_path
        self._reader = GGUFReader(gguf_path)

        if config is None:
            config = self._detect_config()
        self.cfg = config

        # Cache for loaded byte buffers (dense weights only).
        self._cache: dict[str, bytes] = {}

    # ── Public API ──────────────────────────────────────────────────────

    @property
    def gguf_reader(self) -> GGUFReader:
        return self._reader

    def read_tensor_bytes(self, name: str) -> bytes:
        if name not in self._cache:
            raw = self._reader.read_tensor_bytes(name)
            info = self._reader.header.tensors[name]
            # Auto-upcast F16 (ggml_type 1) to F32 so our Python backbone can read it
            if info.ggml_type == 1:
                import numpy as np
                raw = np.frombuffer(raw, dtype=np.float16).astype(np.float32).tobytes()
            self._cache[name] = raw
        return self._cache[name]

    def get_expert_offset(
        self, layer: int, expert: int
    ) -> tuple[int, int, str]:
        """Return ``(offset, size_bytes, dtype)`` for one expert tensor set.

        The expert is identified by its gate-projection tensor; the caller
        is responsible for reading the other two tensors (up, down) at
        contiguous offsets that follow.

        Returns offset/size/dtype for ``blk.{layer}.ffn_gate.{expert}.weight``.
        """
        gate_name = f"blk.{layer}.ffn_gate.{expert}.weight"
        return self._reader.get_tensor_offset(gate_name)

    def get_expert_offsets(
        self, layer: int, expert: int
    ) -> dict[str, tuple[int, int, str]]:
        """Return offset/size/dtype for all three expert tensors.

        Returns dict with keys 'gate', 'up', 'down'.
        """
        return {
            "gate": self._reader.get_tensor_offset(
                f"blk.{layer}.ffn_gate.{expert}.weight"
            ),
            "up": self._reader.get_tensor_offset(
                f"blk.{layer}.ffn_up.{expert}.weight"
            ),
            "down": self._reader.get_tensor_offset(
                f"blk.{layer}.ffn_down.{expert}.weight"
            ),
        }

    # ── Dense weight loading ────────────────────────────────────────────

    def load_embedding(self) -> np.ndarray:
        """Load ``token_embd.weight`` as a float32 array.

        Shape: ``[vocab_size, hidden_dim]``.
        """
        raw = self.read_tensor_bytes("token_embd.weight")
        n = self.cfg.vocab_size * self.cfg.hidden_dim
        arr = np.frombuffer(raw[: n * 4], dtype=np.float32).copy()
        return arr.reshape(self.cfg.vocab_size, self.cfg.hidden_dim)

    def load_output_norm(self) -> RMSNorm:
        """Load ``output_norm.weight`` as an RMSNorm instance."""
        raw = self.read_tensor_bytes("output_norm.weight")
        return RMSNorm(raw, eps=self.cfg.eps)

    def load_lm_head(self) -> LMHead:
        """Load the LM head projection.

        If ``tied_embeddings`` is True (OLMoE default), uses
        ``token_embd.weight``.  Otherwise reads ``output.weight``.
        """
        if self.cfg.tied_embeddings:
            raw = self.read_tensor_bytes("token_embd.weight")
        else:
            raw = self.read_tensor_bytes("output.weight")
        return LMHead(raw, vocab_size=self.cfg.vocab_size,
                      hidden_dim=self.cfg.hidden_dim)

    def load_attention_block(self, layer: int) -> AttentionBlock:
        """Load attention weights for one layer.

        Returns an ``AttentionBlock`` with Q/K/V/O weights pre-loaded.
        """
        return AttentionBlock(
            w_q=self.read_tensor_bytes(f"blk.{layer}.attn_q.weight"),
            w_k=self.read_tensor_bytes(f"blk.{layer}.attn_k.weight"),
            w_v=self.read_tensor_bytes(f"blk.{layer}.attn_v.weight"),
            w_o=self.read_tensor_bytes(f"blk.{layer}.attn_output.weight"),
            num_heads=self.cfg.num_heads,
            head_dim=self.cfg.head_dim,
        )

    def load_attn_norm(self, layer: int) -> RMSNorm:
        """Load pre-attention RMS norm for one layer."""
        raw = self.read_tensor_bytes(f"blk.{layer}.attn_norm.weight")
        return RMSNorm(raw, eps=self.cfg.eps)

    def load_ffn_norm(self, layer: int) -> RMSNorm:
        """Load pre-FFN RMS norm for one layer."""
        raw = self.read_tensor_bytes(f"blk.{layer}.ffn_norm.weight")
        return RMSNorm(raw, eps=self.cfg.eps)

    def load_router(self, layer: int) -> Router:
        """Load the router weight for one MoE layer.

        OLMoE GGUF naming: ``blk.{N}.ffn_gate_inp.weight``.
        """
        raw = self.read_tensor_bytes(f"blk.{layer}.ffn_gate_inp.weight")
        return Router(
            weight=raw,
            num_experts=self.cfg.num_experts,
            top_k=self.cfg.top_k,
            hidden_dim=self.cfg.hidden_dim,
        )

    def load_expert(self, layer: int, expert: int) -> SwiGLUExpert:
        """Load all three weight matrices for one expert into memory.

        Reads and returns a fully-loaded ``SwiGLUExpert``.
        For streaming mode, prefer ``get_expert_offsets()`` instead.
        """
        return SwiGLUExpert(
            w_gate=self.read_tensor_bytes(
                f"blk.{layer}.ffn_gate.{expert}.weight"
            ),
            w_up=self.read_tensor_bytes(
                f"blk.{layer}.ffn_up.{expert}.weight"
            ),
            w_down=self.read_tensor_bytes(
                f"blk.{layer}.ffn_down.{expert}.weight"
            ),
            hidden_dim=self.cfg.hidden_dim,
            intermediate_dim=self.cfg.intermediate_dim,
            quant_type="f32",
        )

    def load_all_dense_weights(self) -> dict[str, Any]:
        """Pre-load all non-expert weights into memory.

        Returns a dict:
            embedding: np.ndarray
            output_norm: RMSNorm
            lm_head: LMHead
            layers: list of dict with keys:
                attn_norm, ffn_norm, router, attention
        """
        result: dict[str, Any] = {
            "embedding": self.load_embedding(),
            "output_norm": self.load_output_norm(),
            "lm_head": self.load_lm_head(),
            "layers": [],
        }

        for layer_idx in range(self.cfg.num_layers):
            layer_weights = {
                "attn_norm": self.load_attn_norm(layer_idx),
                "ffn_norm": self.load_ffn_norm(layer_idx),
                "router": self.load_router(layer_idx),
                "attention": self.load_attention_block(layer_idx),
            }
            result["layers"].append(layer_weights)

        return result

    # ── Config auto-detection ───────────────────────────────────────────

    def _detect_config(self) -> ModelConfig:
        cfg = ModelConfig()

        meta_block_count = self._reader.get_metadata("olmo.block_count")
        if meta_block_count is None:
            meta_block_count = self._reader.get_metadata("llama.block_count")
        if meta_block_count is not None:
            cfg.num_layers = int(meta_block_count)
        else:
            max_layer = -1
            for tname in self._reader.list_tensors():
                if tname.startswith("blk."):
                    try:
                        max_layer = max(max_layer, int(tname.split(".")[1]))
                    except: pass
            if max_layer >= 0:
                cfg.num_layers = max_layer + 1

        t_header = self._reader.header.tensors

        if "token_embd.weight" in t_header:
            dims = t_header["token_embd.weight"].dims
            cfg.vocab_size = dims[0]
            if len(dims) > 1:
                cfg.hidden_dim = dims[1]

        if "blk.0.ffn_gate.0.weight" in t_header:
            cfg.intermediate_dim = t_header["blk.0.ffn_gate.0.weight"].dims[0]

        cfg.tied_embeddings = "output.weight" not in t_header

        if "blk.0.attn_q.weight" in t_header:
            q_rows = t_header["blk.0.attn_q.weight"].dims[0]
            if cfg.hidden_dim > 0:
                for possible_hd in (128, 96, 80, 64):
                    if q_rows % possible_hd == 0 and cfg.hidden_dim % possible_hd == 0:
                        cfg.head_dim = possible_hd
                        cfg.num_heads = q_rows // possible_hd
                        break

        if "blk.0.attn_k.weight" in t_header:
            k_rows = t_header["blk.0.attn_k.weight"].dims[0]
            if cfg.head_dim > 0:
                cfg.num_kv_heads = k_rows // cfg.head_dim

        return cfg

    def __repr__(self) -> str:
        return (
            f"GGUFTensorLoader({os.path.basename(self._path)!r}, "
            f"L={self.cfg.num_layers}, h={self.cfg.hidden_dim}, "
            f"E={self.cfg.num_experts}, V={self.cfg.vocab_size})"
        )

    def close(self) -> None:
        """Clear the tensor cache.  The file handle stays open (GGUFReader)."""
        self._cache.clear()


# ── Self-test ─────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Create a small test GGUF and verify loading."""
    import struct
    import tempfile

    from gguf_stream_reader import GGUFWriter

    print("── gguf_tensor_loader self-test ──")

    # Tiny config for test speed.
    cfg = ModelConfig(
        num_layers=2, hidden_dim=8, intermediate_dim=4,
        num_experts=4, top_k=2, num_heads=2, num_kv_heads=2,
        head_dim=4, vocab_size=16, max_seq_len=64,
    )

    tmp_path = os.path.join(tempfile.gettempdir(),
                            f"_swarm_loader_test_{os.getpid()}.gguf")

    try:
        writer = GGUFWriter(tmp_path)
        writer.add_metadata("general.architecture", "olmo")
        writer.add_metadata("olmo.block_count", cfg.num_layers)

        # Embedding
        emb = np.eye(cfg.vocab_size, cfg.hidden_dim, dtype=np.float32)
        writer.add_tensor("token_embd.weight",
                          [cfg.vocab_size, cfg.hidden_dim],
                          0, emb.tobytes())

        # Output norm
        onorm = np.ones(cfg.hidden_dim, dtype=np.float32)
        writer.add_tensor("output_norm.weight", [cfg.hidden_dim],
                          0, onorm.tobytes())

        # Per-layer weights
        for layer in range(cfg.num_layers):
            # Norms
            writer.add_tensor(f"blk.{layer}.attn_norm.weight",
                              [cfg.hidden_dim], 0, onorm.tobytes())
            writer.add_tensor(f"blk.{layer}.ffn_norm.weight",
                              [cfg.hidden_dim], 0, onorm.tobytes())

            # Attention
            eye = np.eye(cfg.hidden_dim, dtype=np.float32)
            for proj in ("attn_q", "attn_k", "attn_v", "attn_output"):
                writer.add_tensor(f"blk.{layer}.{proj}.weight",
                                  [cfg.hidden_dim, cfg.hidden_dim],
                                  0, eye.tobytes())

            # Router
            router_w = np.zeros((cfg.num_experts, cfg.hidden_dim), dtype=np.float32)
            writer.add_tensor(f"blk.{layer}.ffn_gate_inp.weight",
                              [cfg.num_experts, cfg.hidden_dim],
                              0, router_w.tobytes())

            # Experts
            scale = 0.1 / cfg.intermediate_dim
            for exp in range(cfg.num_experts):
                gate = np.zeros((cfg.intermediate_dim, cfg.hidden_dim), dtype=np.float32)
                gate[:, :cfg.intermediate_dim] = np.eye(cfg.intermediate_dim) * scale
                up = np.zeros((cfg.intermediate_dim, cfg.hidden_dim), dtype=np.float32)
                up[:, :cfg.intermediate_dim] = np.eye(cfg.intermediate_dim) * scale
                down = np.zeros((cfg.hidden_dim, cfg.intermediate_dim), dtype=np.float32)
                down[:cfg.intermediate_dim, :] = np.eye(cfg.intermediate_dim)

                writer.add_tensor(f"blk.{layer}.ffn_gate.{exp}.weight",
                                  [cfg.intermediate_dim, cfg.hidden_dim],
                                  0, gate.tobytes())
                writer.add_tensor(f"blk.{layer}.ffn_up.{exp}.weight",
                                  [cfg.intermediate_dim, cfg.hidden_dim],
                                  0, up.tobytes())
                writer.add_tensor(f"blk.{layer}.ffn_down.{exp}.weight",
                                  [cfg.hidden_dim, cfg.intermediate_dim],
                                  0, down.tobytes())

        writer.write()

        # ── Load back ──────────────────────────────────────────────
        loader = GGUFTensorLoader(tmp_path, config=cfg)
        print(f"  {loader}")

        # Auto-detect
        loader2 = GGUFTensorLoader(tmp_path)
        print(f"  Auto-detected: L={loader2.cfg.num_layers}, "
              f"h={loader2.cfg.hidden_dim}, V={loader2.cfg.vocab_size}")

        assert loader2.cfg.num_layers == 2
        assert loader2.cfg.hidden_dim == 8
        print("  Config detection: ✓")

        # Load embedding
        emb2 = loader.load_embedding()
        assert emb2.shape == (cfg.vocab_size, cfg.hidden_dim)
        assert np.allclose(emb2, emb)
        print(f"  Embedding: {emb2.shape} ✓")

        # Load an attention block
        attn = loader.load_attention_block(0)
        assert attn.num_heads == cfg.num_heads
        print("  Attention block: ✓")

        # Load a router
        router = loader.load_router(0)
        assert router.num_experts == cfg.num_experts
        print("  Router: ✓")

        # Load an expert
        expert = loader.load_expert(0, 0)
        assert expert.hidden_dim == cfg.hidden_dim
        print("  Expert: ✓")

        # Expert offsets
        off = loader.get_expert_offsets(0, 0)
        for k in ("gate", "up", "down"):
            assert k in off, f"Missing {k}"
        print("  Expert offsets: ✓")

        print("── gguf_tensor_loader self-test passed ──")

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    _self_test()
