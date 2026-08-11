#!/usr/bin/env python3
"""
generation_engine.py — Autoregressive MoE generation loop for Swarm.

Ties together the GGUF tensor loader, tokenizer, dense backbone, expert MLP,
and KVCache/RoPE into a complete generation pipeline.  Produces streaming
token output compatible with ``api_server.py``'s SSE interface.

Supports two modes:
  - **Local mode**: all model weights loaded on one node, generation runs
    entirely on that node.  Used for testing and single-node deployments.
  - **Pipeline mode** (future): dispatches layer ranges across a Swarm
    fleet via ``PipelineCoordinator``.

Design rules:
  - Python 3.10+, numpy + existing project modules.
  - ``generate_stream`` is an async generator yielding token strings.
"""

from __future__ import annotations

import asyncio
import logging
import struct
from collections.abc import AsyncIterator
from typing import Any

import numpy as np

from _compat import asyncio_timeout
from dense_backbone import AttentionBlock, KVCache, LMHead, RMSNorm, RoPE
from expert_mlp import Router, SwiGLUExpert
from gguf_stream_reader import GGUFReader
from gguf_tensor_loader import GGUFTensorLoader, ModelConfig
from sharding import ShardAssignment
from tokenizer import SwarmTokenizer

logger = logging.getLogger("swarm.generation_engine")


class GenerationEngine:
    """Autoregressive text generation engine for OLMoE-style MoE models.

    Parameters
    ----------
    gguf_path:
        Path to the GGUF model file.
    tokenizer:
        A ``SwarmTokenizer`` instance for encoding/decoding.
    config:
        Model dimensions.  Auto-detected from GGUF if not provided.
    max_seq_len:
        Maximum sequence length before generation stops (default 2048).
    """

    def __init__(
        self,
        gguf_path: str,
        tokenizer: SwarmTokenizer,
        config: ModelConfig | None = None,
        max_seq_len: int = 2048,
    ) -> None:
        self._loader = GGUFTensorLoader(gguf_path, config=config)
        self.cfg = self._loader.cfg
        self._tokenizer = tokenizer
        self._max_seq_len = max_seq_len

        # ── Pre-load all dense weights ─────────────────────────────────
        logger.info(
            "Loading dense weights from %s (%d layers, h=%d, E=%d)...",
            gguf_path, self.cfg.num_layers, self.cfg.hidden_dim,
            self.cfg.num_experts,
        )
        self._weights = self._loader.load_all_dense_weights()

        # ── RoPE ───────────────────────────────────────────────────────
        self._rope = RoPE(
            head_dim=self.cfg.head_dim,
            max_seq_len=max(self.cfg.max_seq_len, max_seq_len),
            theta=self.cfg.rotary_theta,
        )

        # ── KVCache (created per generation call) ──────────────────────
        self._kv_cache: KVCache | None = None

        logger.info(
            "GenerationEngine ready: %s", self._loader
        )

    # ── Public API ──────────────────────────────────────────────────────

    async def generate_stream(
        self,
        messages: list[dict[str, Any]],
        params: dict[str, Any],
        assignment: ShardAssignment,
    ) -> AsyncIterator[str]:
        """Generate tokens stream from a conversation.

        Yields one token string at a time, compatible with SSE streaming.
        """
        # ── Build prompt ───────────────────────────────────────────────
        prompt = self._build_prompt(messages)
        max_tokens = int(params.get("max_tokens", 256))
        temperature = float(params.get("temperature", 0.7))
        top_p = float(params.get("top_p", 1.0))
        top_k = int(params.get("top_k", 0))

        # ── Tokenize ───────────────────────────────────────────────────
        input_ids = self._tokenizer.encode(prompt)
        if not input_ids:
            logger.warning("Empty prompt after tokenization")
            yield ""
            return

        # ── Add BOS if not present ─────────────────────────────────────
        bos = self._tokenizer.bos_token_id
        if bos is not None and (not input_ids or input_ids[0] != bos):
            input_ids = [bos] + input_ids

        logger.debug("Prompt: %d tokens", len(input_ids))

        # ── Reset KVCache ──────────────────────────────────────────────
        self._kv_cache = KVCache(
            num_layers=self.cfg.num_layers,
            num_kv_heads=self.cfg.num_kv_heads,
            head_dim=self.cfg.head_dim,
            max_seq_len=self._max_seq_len,
        )

        # ── Prefill ────────────────────────────────────────────────────
        logits: np.ndarray | None = None
        for pos, token_id in enumerate(input_ids):
            logits = self._forward_one_token(token_id, pos)

        # Sample the first output token.
        assert logits is not None
        next_token_id: int | None = self._sample(
            logits, temperature, top_p, top_k
        )

        # ── Decode loop ────────────────────────────────────────────────
        eos = self._tokenizer.eos_token_id
        tokens_generated = 0
        while tokens_generated < max_tokens and next_token_id is not None:
            # Yield the current token.
            tokens_generated += 1
            token_text = self._tokenizer.decode([next_token_id])
            yield token_text

            # Stop conditions.
            if eos is not None and next_token_id == eos:
                break
            if len(input_ids) + tokens_generated >= self._max_seq_len:
                break

            # Compute next token.
            pos = len(input_ids) + tokens_generated - 1
            logits = self._forward_one_token(next_token_id, pos)
            next_token_id = self._sample(logits, temperature, top_p, top_k)

    # ── Internal: prompt building ───────────────────────────────────────

    def _build_prompt(self, messages: list[dict[str, Any]]) -> str:
        """Convert a messages list to a prompt string.

        Uses a simple format: ``<|user|>text<|assistant|>``
        """
        parts: list[str] = []
        for msg in messages:
            role = msg.get("role", "user")
            content = str(msg.get("content", ""))
            if role == "system":
                parts.append(f"<|system|>\n{content}\n")
            elif role == "user":
                parts.append(f"<|user|>\n{content}\n")
            elif role == "assistant":
                parts.append(f"<|assistant|>\n{content}\n")
        parts.append("<|assistant|>\n")
        return "\n".join(parts)

    # ── Internal: forward pass ──────────────────────────────────────────

    def _forward_one_token(self, token_id: int, position: int) -> np.ndarray:
        """Run one token through all layers.

        Returns the raw logits vector (before sampling).

        Parameters
        ----------
        token_id:
            The integer token ID to process.
        position:
            Absolute position in the sequence (0-based).

        Returns
        -------
        np.ndarray
            Logits vector of shape ``[vocab_size]``.
        """
        assert self._kv_cache is not None

        # ── Embedding lookup ──────────────────────────────────────────
        emb = self._weights["embedding"]
        x = emb[token_id].astype(np.float32).copy()  # [hidden_dim]

        # ── Layers ────────────────────────────────────────────────────
        for layer_idx in range(self.cfg.num_layers):
            layer = self._weights["layers"][layer_idx]
            attn_norm: RMSNorm = layer["attn_norm"]
            ffn_norm: RMSNorm = layer["ffn_norm"]
            attention: AttentionBlock = layer["attention"]
            router: Router = layer["router"]

            # ── Attention block ─────────────────────────────────
            attn_in = attn_norm.forward(x)
            attn_out = attention.forward(
                attn_in, self._rope, self._kv_cache,
                layer_idx, position,
            )
            x = x + attn_out  # residual

            # ── MoE FFN block ───────────────────────────────────
            ffn_in = ffn_norm.forward(x)
            expert_indices, expert_weights = router.forward(ffn_in)

            # Evaluate selected experts.
            expert_outputs = []
            for exp_idx in expert_indices:
                expert = self._get_expert(layer_idx, exp_idx)
                expert_out = expert.forward(ffn_in)
                expert_outputs.append(expert_out)

            # Weighted sum.
            moe_out = np.zeros(self.cfg.hidden_dim, dtype=np.float32)
            for weight, exp_out in zip(expert_weights, expert_outputs):
                moe_out += weight * exp_out

            x = x + moe_out  # residual

        # Advance KVCache after all layers.
        self._kv_cache.advance()

        # ── Output norm + LM head ─────────────────────────────────────
        out_norm: RMSNorm = self._weights["output_norm"]
        lm_head: LMHead = self._weights["lm_head"]
        x = out_norm.forward(x)
        logits = lm_head.forward(x)  # [vocab_size]

        return logits

    def _get_expert(self, layer_idx: int, expert_idx: int) -> SwiGLUExpert:
        """Get or load an expert for a specific layer.

        Experts are loaded lazily and cached.  In a real distributed
        deployment this would dispatch to the owning node.
        """
        # Cache key
        cache_key = f"expert_{layer_idx}_{expert_idx}"
        if not hasattr(self, "_expert_cache"):
            self._expert_cache: dict[str, SwiGLUExpert] = {}

        if cache_key not in self._expert_cache:
            self._expert_cache[cache_key] = self._loader.load_expert(
                layer_idx, expert_idx
            )

        return self._expert_cache[cache_key]

    # ── Internal: sampling ──────────────────────────────────────────────

    def _sample(
        self,
        logits: np.ndarray,
        temperature: float,
        top_p: float,
        top_k: int,
    ) -> int:
        """Sample the next token ID from logits.

        Supports: greedy (temperature=0), temperature scaling, top-k,
        top-p (nucleus) sampling.
        """
        if temperature <= 0:
            return int(np.argmax(logits))

        # Temperature scaling.
        scaled = logits / temperature

        # Top-k.
        if top_k > 0:
            k = min(top_k, len(scaled))
            indices = np.argpartition(-scaled, k - 1)[:k]
            mask = np.ones_like(scaled, dtype=bool)
            mask[indices] = False
            scaled[mask] = -np.inf

        # Top-p (nucleus).
        if top_p < 1.0:
            sorted_indices = np.argsort(-scaled)
            sorted_logits = scaled[sorted_indices]
            probs = self._softmax(sorted_logits)
            cumsum = np.cumsum(probs)
            cutoff_idx = int(np.searchsorted(cumsum, top_p, side="right"))
            if cutoff_idx < len(cumsum):
                sorted_logits[cutoff_idx + 1:] = -np.inf
            scaled = np.full_like(scaled, -np.inf)
            scaled[sorted_indices] = sorted_logits

        # Softmax + sample.
        probs = self._softmax(scaled)
        # If all -inf (shouldn't happen), fall back to argmax.
        if np.all(probs == 0):
            return int(np.argmax(logits))

        return int(np.random.choice(len(probs), p=probs))

    @staticmethod
    def _softmax(x: np.ndarray) -> np.ndarray:
        """Numerically stable softmax."""
        x_max = np.max(x)
        e_x = np.exp(x - x_max)
        return e_x / np.sum(e_x)


# ── Pipeline-mode compute stage factory ───────────────────────────────────


def make_pipeline_compute_stage(
    loader: GGUFTensorLoader,
    layer_start: int,
    layer_end: int,
    kv_cache: KVCache,
    rope: RoPE,
) -> Any:
    """Create a ``compute_stage`` callback for a pipeline node.

    The callback runs layers [layer_start, layer_end) and returns the
    serialized output activation.

    Parameters
    ----------
    loader:
        The GGUF tensor loader (must have pre-loaded all dense weights).
    layer_start:
        First layer index to compute (inclusive).
    layer_end:
        Last layer index to compute (exclusive).
    kv_cache:
        Shared KVCache instance.
    rope:
        Shared RoPE instance.
    """
    # Pre-load all weights for this layer range.
    weights = loader.load_all_dense_weights()
    all_layers = weights["layers"]
    my_layers = all_layers[layer_start:layer_end]

    def compute_stage(
        activation_bytes: bytes,
        _layer_start: int,
        _layer_end: int,
    ) -> bytes:
        """Run the layer range on the activation.

        The activation bytes encode: [4B: n_floats][n_floats * 4B: f32 array]
        """
        # Decode activation.
        n_floats = struct.unpack("<I", activation_bytes[:4])[0]
        x = np.frombuffer(
            activation_bytes[4: 4 + n_floats * 4], dtype=np.float32
        ).copy()

        hidden_dim = len(x)
        position = kv_cache.seq_len

        for local_layer_idx in range(len(my_layers)):
            global_layer_idx = layer_start + local_layer_idx
            layer = my_layers[local_layer_idx]

            attn_norm: RMSNorm = layer["attn_norm"]
            ffn_norm: RMSNorm = layer["ffn_norm"]
            attention: AttentionBlock = layer["attention"]
            router: Router = layer["router"]

            # Attention.
            attn_in = attn_norm.forward(x)
            attn_out = attention.forward(
                attn_in, rope, kv_cache, global_layer_idx, position,
            )
            x = x + attn_out

            # MoE FFN.
            ffn_in = ffn_norm.forward(x)
            expert_indices, expert_weights = router.forward(ffn_in)

            moe_out = np.zeros(hidden_dim, dtype=np.float32)
            for exp_idx, exp_weight in zip(expert_indices, expert_weights):
                expert = loader.load_expert(global_layer_idx, exp_idx)
                exp_output = expert.forward(ffn_in)
                moe_out += exp_weight * exp_output

            x = x + moe_out

        # Encode output.
        out_floats = x.astype(np.float32).tobytes()
        return struct.pack("<I", hidden_dim) + out_floats

    return compute_stage


# ── Self-test ─────────────────────────────────────────────────────────────


def _self_test() -> None:
    """Smoke test with a tiny model and character tokenizer."""
    import os
    import tempfile

    from gguf_stream_reader import GGUFWriter
    from tokenizer import make_char_tokenizer

    print("── generation_engine self-test ──")

    cfg = ModelConfig(
        num_layers=2, hidden_dim=8, intermediate_dim=4,
        num_experts=4, top_k=2, num_heads=2, num_kv_heads=2,
        head_dim=4, vocab_size=16, max_seq_len=32,
        tied_embeddings=True,
    )

    tmp_path = os.path.join(
        tempfile.gettempdir(),
        f"_swarm_gen_test_{os.getpid()}.gguf",
    )

    try:
        # Build test GGUF.
        writer = GGUFWriter(tmp_path)
        writer.add_metadata("general.architecture", "olmo")

        emb = np.eye(cfg.vocab_size, cfg.hidden_dim, dtype=np.float32)
        writer.add_tensor("token_embd.weight",
                          [cfg.vocab_size, cfg.hidden_dim],
                          0, emb.tobytes())

        onorm = np.ones(cfg.hidden_dim, dtype=np.float32)
        writer.add_tensor("output_norm.weight", [cfg.hidden_dim],
                          0, onorm.tobytes())

        for layer in range(cfg.num_layers):
            writer.add_tensor(f"blk.{layer}.attn_norm.weight",
                              [cfg.hidden_dim], 0, onorm.tobytes())
            writer.add_tensor(f"blk.{layer}.ffn_norm.weight",
                              [cfg.hidden_dim], 0, onorm.tobytes())

            eye = np.eye(cfg.hidden_dim, dtype=np.float32) * 0.1
            for proj in ("attn_q", "attn_k", "attn_v", "attn_output"):
                writer.add_tensor(f"blk.{layer}.{proj}.weight",
                                  [cfg.hidden_dim, cfg.hidden_dim],
                                  0, eye.tobytes())

            router_w = (np.random.RandomState(42 + layer)
                        .randn(cfg.num_experts, cfg.hidden_dim)
                        .astype(np.float32) * 0.01)
            writer.add_tensor(f"blk.{layer}.ffn_gate_inp.weight",
                              [cfg.num_experts, cfg.hidden_dim],
                              0, router_w.tobytes())

            scale = 0.1 / cfg.intermediate_dim
            for exp in range(cfg.num_experts):
                gate = np.zeros((cfg.intermediate_dim, cfg.hidden_dim),
                                dtype=np.float32)
                gate[:, :cfg.intermediate_dim] = (
                    np.eye(cfg.intermediate_dim) * scale
                )
                up = np.zeros_like(gate)
                up[:, :cfg.intermediate_dim] = (
                    np.eye(cfg.intermediate_dim) * scale
                )
                down = np.zeros((cfg.hidden_dim, cfg.intermediate_dim),
                                dtype=np.float32)
                down[:cfg.intermediate_dim, :] = (
                    np.eye(cfg.intermediate_dim)
                )
                writer.add_tensor(
                    f"blk.{layer}.ffn_gate.{exp}.weight",
                    [cfg.intermediate_dim, cfg.hidden_dim],
                    0, gate.tobytes(),
                )
                writer.add_tensor(
                    f"blk.{layer}.ffn_up.{exp}.weight",
                    [cfg.intermediate_dim, cfg.hidden_dim],
                    0, up.tobytes(),
                )
                writer.add_tensor(
                    f"blk.{layer}.ffn_down.{exp}.weight",
                    [cfg.hidden_dim, cfg.intermediate_dim],
                    0, down.tobytes(),
                )

        writer.write()

        # Build tokenizer and engine.
        # Use only the first cfg.vocab_size - 4 chars so token IDs
        # stay within the embedding table bounds.
        safe_chars = "".join(chr(i) for i in range(32, 32 + cfg.vocab_size - 4))
        tok = make_char_tokenizer(vocab_chars=safe_chars)
        engine = GenerationEngine(tmp_path, tok, config=cfg)

        # Run generation.
        from sharding import ShardAssignment

        dummy_assignment = ShardAssignment(
            node_experts={"self": []},
            node_counts={"self": 0},
            node_bandwidths={"self": 0},
            fleet_hash="0" * 64,
            num_experts=cfg.num_experts,
            total_bandwidth_mbps=0,
        )

        async def _run():
            tokens = []
            async for token_text in engine.generate_stream(
                messages=[{"role": "user", "content": "Hi"}],
                params={"max_tokens": 4, "temperature": 0.7},
                assignment=dummy_assignment,
            ):
                tokens.append(token_text)
            return tokens

        tokens = asyncio.run(_run())
        print(f"  Generated {len(tokens)} tokens: {tokens}")
        assert len(tokens) > 0, "No tokens generated"
        assert len(tokens) <= 4, f"Too many tokens: {len(tokens)}"
        print(f"  Output: {''.join(tokens)!r} ✓")

        print("── generation_engine self-test passed ──")

    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


if __name__ == "__main__":
    _self_test()
