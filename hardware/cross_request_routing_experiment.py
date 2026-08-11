#!/usr/bin/env python3
"""
Cross-request routing-correlation experiment for a streaming MoE.

Question: When OLMoE-1B-7B generates tokens for N independent, unrelated
requests (one token each at the same generation position), how much do the
routed experts overlap across those requests?

This is different from the consecutive-token experiment
(speculative_routing_experiment.py), which measures overlap between
consecutive tokens from ONE continuous stream.  Consecutive tokens share
context and are therefore correlated.  Cross-request tokens share nothing
and should correlate less — possibly not at all.

Why this matters
----------------
For speculative decoding, the economics are:

    Tokens gained       Byte cost (union multiplier)   Net
    ~2.2 (acceptance)   2.20–2.59× (consecutive)       LOSS

For request batching, the economics are the reverse:

    Tokens gained per batch    Byte cost (union multiplier)    Net
    4 (one per request)        unknown — this experiment       ?

If cross-request union lands near the consecutive figure (~2.4×), batching
wins ~1.6× and a batch scheduler is worth building.  If it lands near the
independent prediction (3.81×), batching wins ~1.05× and is barely worth
anything — in which case the right scheduler is "always batch 1".

Method
------
1. Take N independent prompts (default 8), covering genuinely different
   topics.  Independence is the experimental variable.
2. For each prompt independently, generate --tokens-per-prompt tokens
   (default 200) with router hooks capturing expert selections.
3. Build cross-request windows: for window size B, take one token from
   each of B different prompt traces at the same generation position.
   Compute the union of their selected experts.  Slide over all positions
   and over all combinations of B distinct prompts.
4. Compute the consecutive condition from the same traces, so both
   conditions are measured on identical model state and identical seeds.

Model: OLMoE-1B-7B (16 layers, 64 experts, top-8 routing, dropless).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, OlmoeForCausalLM

try:
    from transformers import BitsAndBytesConfig
    _HAS_BNB = True
except ImportError:
    _HAS_BNB = False

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
NUM_EXPERTS = 64
TOP_K = 8
NUM_LAYERS = 16
WINDOW_SIZES = [2, 4, 8]

# Eight genuinely unrelated prompts covering different topics.
# Independence is the entire experimental variable — these must not be
# variations on one theme.
DEFAULT_PROMPTS = [
    # Cooking
    "Write a detailed recipe for Thai green curry with chicken, including "
    "the ingredient list, preparation steps, and cooking times. Explain "
    "why certain techniques (like blooming the curry paste in oil) matter.",

    # Programming / debugging
    "I have a Python script that reads a large CSV file with pandas and "
    "computes group-by aggregations. It runs fine on 100 MB files but "
    "hangs indefinitely on a 2 GB file — CPU sits at 100% on one core "
    "and memory usage keeps climbing. What is likely wrong and how do I "
    "fix it?",

    # History
    "Explain the causes of the Peloponnesian War between Athens and Sparta. "
    "Cover the immediate trigger (the dispute over Corcyra and Potidaea), "
    "the underlying structural causes (Athenian imperial expansion, Spartan "
    "fear of growing Athenian power), and how Thucydides' distinction "
    "between aitiai and prophasis applies.",

    # Physics
    "Derive the Schwarzschild radius from Newtonian gravity by setting the "
    "escape velocity equal to the speed of light. Then explain why, "
    "coincidentally, the general-relativistic derivation produces exactly "
    "the same formula, and what this tells us (and doesn't tell us) about "
    "the relationship between Newtonian and relativistic gravity.",

    # Literary analysis
    "Analyze the opening paragraph of Gabriel García Márquez's 'One Hundred "
    "Years of Solitude.' Discuss the narrative technique of prolepsis (the "
    "colonel remembering the ice), the tension between cyclical and linear "
    "time, and how these three sentences establish the novel's central "
    "themes of memory, fate, and the blurring of myth and history.",

    # Mathematics
    "Prove that the square root of 2 is irrational using a proof by "
    "contradiction. Start by assuming sqrt(2) = a/b where a and b are "
    "coprime integers, then derive a contradiction. After the proof, "
    "explain why this result was so disturbing to the Pythagoreans and "
    "what it reveals about the limitations of rational numbers.",

    # Biology
    "Describe the mechanism of CRISPR-Cas9 gene editing in detail. Cover: "
    "how the guide RNA directs Cas9 to the target sequence, the role of "
    "the PAM sequence, how the double-strand break is repaired via NHEJ "
    "or HDR, and two current therapeutic applications with their specific "
    "molecular targets.",

    # Economics
    "Explain the concept of comparative advantage in international trade, "
    "using Ricardo's original example of England and Portugal trading cloth "
    "and wine. Show the numerical example step by step, explain why both "
    "countries gain from trade even when one is more productive at "
    "everything, and discuss one modern criticism of the model.",
    # Music theory
    "Explain why the circle of fifths works the way it does, starting from "
    "the physics of overtones. Cover why twelve semitones divide the octave, "
    "what equal temperament sacrifices, and why a perfect fifth sounds "
    "consonant.",

    # Geology
    "Describe how a subduction zone produces both volcanoes and deep ocean "
    "trenches. Cover the density differences between oceanic and continental "
    "crust, the role of water in lowering the melting point of mantle rock, "
    "and why the volcanoes appear inland rather than at the trench.",

    # Law
    "Explain the difference between civil law and common law legal systems. "
    "Cover the role of judicial precedent, how statutes are interpreted "
    "differently, and give an example of how the same dispute might be "
    "resolved differently under each.",

    # Sports / biomechanics
    "Explain the biomechanics of a baseball pitcher's throwing motion. Cover "
    "the kinetic chain from the legs through the hips and torso to the arm, "
    "where the energy actually comes from, and why shoulder and elbow "
    "injuries are so common.",

    # Cartography
    "Explain why every flat map of the Earth must distort something. Cover "
    "Gauss's Theorema Egregium in accessible terms, compare what Mercator "
    "and Gall-Peters each preserve and sacrifice, and explain why there is "
    "no single best projection.",

    # Agriculture
    "Explain crop rotation and why it works. Cover nitrogen fixation by "
    "legumes, how rotation interrupts pest and pathogen life cycles, and why "
    "monoculture depletes soil in ways that fertiliser does not fully fix.",

    # Typography
    "Explain the difference between serif and sans-serif typefaces, why "
    "serifs existed historically, and how the constraints of low-resolution "
    "screens changed typeface design. Cover hinting and why it mattered.",

    # Immunology
    "Explain how vaccines produce immunological memory. Cover the difference "
    "between B cells and T cells, what an adjuvant does, and why some "
    "vaccines need boosters while others confer lifelong immunity."
]


# ── Hook machinery ─────────────────────────────────────────────────────────────
# Identical to speculative_routing_experiment.py — do not diverge.

def install_router_hooks(model: OlmoeForCausalLM) -> dict:
    """Register forward hooks on every router in the model.

    Returns a dict to be populated: captured[layer_idx].append(selected_experts_array)
    """
    captured = defaultdict(list)

    def make_hook(layer_idx: int):
        def hook(module, input_, output):
            # OlmoeTopKRouter returns (router_logits, router_scores, router_indices)
            # router_indices: (batch*seq_len, top_k) LongTensor of expert indices
            # output[2] is indices — output[1] is scores (this was wrong once before)
            selected_experts = output[2].detach().cpu().numpy().astype(np.int16)
            captured[layer_idx].append(selected_experts)

        return hook

    for i in range(NUM_LAYERS):
        router = model.model.layers[i].mlp.gate
        router.register_forward_hook(make_hook(i))

    return captured


# ── Independent-routing prediction ─────────────────────────────────────────────

def predicted_union(e: int, k: int, b: int) -> float:
    """Expected distinct experts when B tokens independently select k from e."""
    return e * (1.0 - (1.0 - k / e) ** b)


# ── Generation per prompt ──────────────────────────────────────────────────────

def generate_one_prompt(
    model: OlmoeForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    prompt_index: int,
    max_new_tokens: int,
    temperature: float,
    base_seed: int,
    force_full_length: bool = False,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Generate tokens for one prompt and capture expert selections.

    Each prompt gets its own seed (base_seed + prompt_index) so runs are
    reproducible individually and across a sweep.

    Returns:
        generated_ids: token IDs of the generated portion (excluding prompt).
        expert_trace: dict mapping layer_idx -> array of shape (n_gen_tokens, top_k).
    """
    seed = base_seed + prompt_index
    torch.manual_seed(seed)

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    prompt_len = inputs.input_ids.shape[1]

    captured = install_router_hooks(model)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=None if force_full_length else tokenizer.eos_token_id,
            min_new_tokens=max_new_tokens if force_full_length else None,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0, prompt_len:].tolist()
    n_gen = len(generated_ids)

    # Assemble expert trace — same logic as the original script.
    expert_trace = {}
    for layer_idx in range(NUM_LAYERS):
        chunks = captured[layer_idx]
        if not chunks:
            raise RuntimeError(f"No router outputs captured for layer {layer_idx}")

        gen_chunks = []
        for i, ch in enumerate(chunks):
            if i == 0:
                if ch.shape[0] > prompt_len:
                    gen_chunks.append(ch[prompt_len:])
            else:
                gen_chunks.append(ch)

        if not gen_chunks:
            raise RuntimeError(
                f"No generated-token router outputs for layer {layer_idx}."
            )

        trace = np.concatenate(gen_chunks, axis=0)
        if trace.shape[0] != n_gen:
            if trace.shape[0] > n_gen:
                trace = trace[:n_gen]
            else:
                pad = np.full((n_gen - trace.shape[0], TOP_K), -1, dtype=np.int16)
                trace = np.concatenate([trace, pad], axis=0)

        expert_trace[layer_idx] = trace

    return generated_ids, expert_trace


# ── Consecutive-token analysis (same-prompt sliding windows) ────────────────────

def analyze_consecutive(
    expert_traces: list[dict[int, np.ndarray]],
    min_tokens: int,
) -> dict:
    """Compute actual vs predicted union for consecutive tokens within each prompt.

    For each prompt independently, runs the standard sliding-window analysis
    (identical to speculative_routing_experiment.py).  Results are averaged
    across prompts.

    Returns the same shape as analyze_routing_correlation() in the original
    script: {per_layer, overall, popularity}.
    """
    per_layer_accum: dict[int, dict[int, list[float]]] = {
        li: {B: [] for B in WINDOW_SIZES} for li in range(NUM_LAYERS)
    }
    expert_counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)

    for trace in expert_traces:
        # Each prompt contributes everything it generated; no global clamp
        # (see the note in analyze_cross_request).
        usable = next(iter(trace.values())).shape[0]

        for layer_idx in range(NUM_LAYERS):
            t = trace[layer_idx][:usable]  # (usable, top_k)

            for B in WINDOW_SIZES:
                if usable < B:
                    continue
                for start in range(usable - B + 1):
                    window = t[start : start + B]
                    union_size = len(set(window.flatten().tolist()))
                    per_layer_accum[layer_idx][B].append(union_size)

            # Tally expert selections
            for pos in range(usable):
                for e in t[pos]:
                    expert_counts[layer_idx, e] += 1

    # Build results
    per_layer = {}
    for layer_idx in range(NUM_LAYERS):
        layer_results = {}
        for B in WINDOW_SIZES:
            vals = per_layer_accum[layer_idx][B]
            if vals:
                actual_mean = np.mean(vals)
            else:
                actual_mean = np.nan
            pred = predicted_union(NUM_EXPERTS, TOP_K, B)
            ratio = actual_mean / pred if not np.isnan(actual_mean) else np.nan
            layer_results[B] = (actual_mean, pred, ratio)
        per_layer[layer_idx] = layer_results

    overall = {}
    for B in WINDOW_SIZES:
        actuals = []
        for layer_idx in range(NUM_LAYERS):
            a, _, _ = per_layer[layer_idx][B]
            if not np.isnan(a):
                actuals.append(a)
        pred = predicted_union(NUM_EXPERTS, TOP_K, B)
        overall[B] = (
            np.mean(actuals) if actuals else np.nan,
            pred,
            (np.mean(actuals) / pred) if actuals else np.nan,
        )

    popularity = {}
    for layer_idx in range(NUM_LAYERS):
        counts = expert_counts[layer_idx]
        total = counts.sum()
        top_5pct_n = max(1, int(np.ceil(NUM_EXPERTS * 0.05)))
        sorted_desc = np.sort(counts)[::-1]
        top_5pct_count = sorted_desc[:top_5pct_n].sum()
        popularity[layer_idx] = {
            "top_5pct_fraction": (
                round(top_5pct_count / total, 4) if total > 0 else 0
            ),
            "top_5pct_experts": [
                int(e) for e in np.argsort(counts)[::-1][:top_5pct_n]
            ],
            "expert_counts": counts.tolist(),
        }

    return {"per_layer": per_layer, "overall": overall, "popularity": popularity}


# ── Cross-request analysis ─────────────────────────────────────────────────────

def analyze_cross_request(
    expert_traces: list[dict[int, np.ndarray]],
    num_prompts: int,
    min_tokens: int,
) -> dict:
    """Compute actual vs predicted union for cross-request token windows.

    For window size B, takes one token from each of B different prompt traces
    at the same generation position.  Uses all C(N, B) combinations of prompts
    exhaustively (deterministic, no sampling needed at N=8, B≤8).

    Returns the same shape: {per_layer, overall}.
    """
    per_layer_accum: dict[int, dict[int, list[float]]] = {
        li: {B: [] for B in WINDOW_SIZES} for li in range(NUM_LAYERS)
    }

    for B in WINDOW_SIZES:
        if B > num_prompts:
            continue

        # All combinations of B distinct prompts.
        for combo in combinations(range(num_prompts), B):
            traces_in_combo = [expert_traces[i] for i in combo]
            # Each trace might have a different length; use the minimum.
            # Use only what THIS combination actually has, not the global
            # minimum across all prompts.  Clamping to the global min means a
            # single short prompt (one that hit EOS early) truncates every
            # other prompt too -- in one run a 31-token prompt threw away
            # ~85% of the generated data.  Combinations end up with unequal
            # sample counts, which is fine: the mean is taken over all
            # samples, not over per-combination means.
            usable = min(
                next(iter(t.values())).shape[0] for t in traces_in_combo
            )

            for pos in range(usable):
                for layer_idx in range(NUM_LAYERS):
                    union_set: set[int] = set()
                    for t in traces_in_combo:
                        for e in t[layer_idx][pos]:
                            union_set.add(int(e))
                    union_size = len(union_set)
                    per_layer_accum[layer_idx][B].append(union_size)

    per_layer = {}
    for layer_idx in range(NUM_LAYERS):
        layer_results = {}
        for B in WINDOW_SIZES:
            vals = per_layer_accum[layer_idx][B]
            if vals:
                actual_mean = np.mean(vals)
            else:
                actual_mean = np.nan
            pred = predicted_union(NUM_EXPERTS, TOP_K, B)
            ratio = actual_mean / pred if not np.isnan(actual_mean) else np.nan
            layer_results[B] = (actual_mean, pred, ratio)
        per_layer[layer_idx] = layer_results

    overall = {}
    for B in WINDOW_SIZES:
        actuals = []
        for layer_idx in range(NUM_LAYERS):
            a, _, _ = per_layer[layer_idx][B]
            if not np.isnan(a):
                actuals.append(a)
        pred = predicted_union(NUM_EXPERTS, TOP_K, B)
        overall[B] = (
            np.mean(actuals) if actuals else np.nan,
            pred,
            (np.mean(actuals) / pred) if actuals else np.nan,
        )

    return {"per_layer": per_layer, "overall": overall}


# ── Output ─────────────────────────────────────────────────────────────────────

def print_report(
    results_consecutive: dict,
    results_cross: dict,
    num_prompts: int,
    tokens_per_prompt: int,
    min_tokens: int,
    out_dir: str,
    quantized: bool = False,
    seed: int = 42,
):
    """Print the side-by-side comparison and the batching verdict."""
    overall_c = results_consecutive["overall"]
    overall_x = results_cross["overall"]

    sep = "─" * 90
    lines: list[str] = []

    if quantized:
        lines.append(
            "NOTE: routing decisions were traced on an 8-bit-quantized model, "
            "not full precision. Router argmax is a discrete choice and mostly "
            "insulated from main-model quantization, but this is a real "
            "(second-order) deviation from measuring the model as published. "
            "Treat results as directional evidence."
        )
        lines.append("")

    lines.append(sep)
    lines.append("  Cross-Request vs Consecutive Routing Correlation")
    lines.append(sep)
    lines.append(
        f"  Prompts: {num_prompts}, tokens per prompt: {tokens_per_prompt}, "
        f"min usable: {min_tokens}"
    )
    lines.append(f"  Seed: {seed}")
    lines.append(sep)
    lines.append("")

    # ── Per-layer detail table ──────────────────────────────────────────
    # Print the full per-layer table for the cross-request condition first,
    # since that's the novel measurement.
    for condition_name, results in [
        ("CROSS-REQUEST (independent prompts)", results_cross),
        ("CONSECUTIVE (same stream)", results_consecutive),
    ]:
        per_layer = results["per_layer"]
        lines.append(f"  ── {condition_name} ──")
        header = f"  {'Layer':>6}"
        for B in WINDOW_SIZES:
            header += f"  B={B} actual  pred  ratio"
        lines.append(header)
        lines.append("  " + "-" * (len(header) - 2))

        for layer_idx in range(NUM_LAYERS):
            row = f"  {layer_idx:>6}"
            for B in WINDOW_SIZES:
                actual, pred, ratio = per_layer[layer_idx][B]
                if np.isnan(actual):
                    row += f"  {'':>6}  {pred:5.1f}  {'':>5}"
                else:
                    row += f"  {actual:6.2f}  {pred:5.1f}  {ratio:.3f}"
            lines.append(row)

        # Overall row
        row = f"  {'OVERALL':>6}"
        for B in WINDOW_SIZES:
            actual, pred, ratio = results["overall"][B]
            if np.isnan(actual):
                row += f"  {'':>6}  {pred:5.1f}  {'':>5}"
            else:
                row += f"  {actual:6.2f}  {pred:5.1f}  {ratio:.3f}"
        lines.append(row)
        lines.append("")

    # ── Side-by-side summary ────────────────────────────────────────────
    lines.append(sep)
    lines.append("  Side-by-Side Summary")
    lines.append(sep)
    summary_header = (
        f"  {'Condition':<32} {'B':>3}  {'Actual':>7}  {'Pred':>6}  "
        f"{'Ratio':>6}  {'Byte Mult':>9}"
    )
    lines.append(summary_header)
    lines.append("  " + "-" * (len(summary_header) - 2))

    for B in WINDOW_SIZES:
        # Consecutive
        ac, pc, rc = overall_c[B]
        bm_c = ac / TOP_K if not np.isnan(ac) else float("nan")
        lines.append(
            f"  {'Consecutive (same stream)':<32} {B:>3}  "
            f"{ac:7.2f}  {pc:6.1f}  {rc:6.3f}  {bm_c:9.2f}x"
        )

        # Cross-request
        ax, px, rx = overall_x[B]
        bm_x = ax / TOP_K if not np.isnan(ax) else float("nan")
        lines.append(
            f"  {'Cross-request (independent)':<32} {B:>3}  "
            f"{ax:7.2f}  {px:6.1f}  {rx:6.3f}  {bm_x:9.2f}x"
        )

        # Independent-sampling theory
        pi = predicted_union(NUM_EXPERTS, TOP_K, B)
        bm_i = pi / TOP_K
        lines.append(
            f"  {'Independent-sampling theory':<32} {B:>3}  "
            f"{'—':>7}  {pi:6.1f}  {'1.000':>6}  {bm_i:9.2f}x"
        )
        lines.append("")

    # ── Batching verdict ────────────────────────────────────────────────
    lines.append(sep)
    lines.append("  Batching Verdict")
    lines.append(sep)
    lines.append(
        "  For batching: each request contributes 1 token, so B requests "
        "produce B tokens."
    )
    lines.append(
        "  Benefit = B / cross_request_byte_multiplier."
    )
    lines.append("")

    any_pays = False
    for B in WINDOW_SIZES:
        ax, _, _ = overall_x[B]
        if np.isnan(ax):
            continue
        byte_mult = ax / TOP_K
        gain = B / byte_mult
        # bool() is load-bearing — numpy.bool_ is not Python True/False
        pays = bool(gain > 1.0)

        verdict = (
            f"BATCHING PAYS ({gain:.2f}x net gain)"
            if pays
            else f"BATCHING IS A NET LOSS ({gain:.2f}x net — below 1.0x)"
        )

        lines.append(
            f"  B={B}: {B} tokens for {byte_mult:.2f}x bytes "
            f"= {gain:.2f}x net gain — {verdict}"
        )
        if pays:
            any_pays = True

    lines.append("")

    # ── Overall recommendation ──────────────────────────────────────────
    # Check B=4 specifically since that's the most practical batch size.
    b4_ax, _, _ = overall_x[4]
    if not np.isnan(b4_ax):
        b4_gain = 4 / (b4_ax / TOP_K)
        if bool(b4_gain > 1.3):
            lines.append(
                "  RECOMMENDATION: Batching pays at B=4. A batch scheduler "
                "that groups independent requests is worth building — it "
                "improves throughput by {:.2f}x over batch-1 at the same "
                "per-user latency cost as the byte multiplier.".format(b4_gain)
            )
        elif bool(b4_gain > 1.05):
            lines.append(
                "  RECOMMENDATION: Batching shows mild gain at B=4 ({:.2f}x). "
                "A batch scheduler helps but is not transformative. The "
                "simpler 'always batch 1' design is defensible.".format(b4_gain)
            )
        else:
            lines.append(
                "  RECOMMENDATION: Batching does not pay at B=4 ({:.2f}x net, "
                "below or barely above 1.0x). The right scheduler is "
                "'always batch 1' — far simpler and loses almost nothing.".format(b4_gain)
            )
    lines.append("")

    # ── Methodological note ─────────────────────────────────────────────
    lines.append(sep)
    lines.append("  Methodological Notes")
    lines.append(sep)
    lines.append(
        "  • Each prompt was generated independently via its own model.generate()"
    )
    lines.append(
        "    call. The model is in eval() mode; no state persists between calls."
    )
    lines.append(
        "    Each call gets its own seed (base_seed + prompt_index), and hooks"
    )
    lines.append(
        "    are reinstalled fresh per call. Sequential generation does not bias"
    )
    lines.append(
        "    the routing traces — each prompt's tokens are generated from"
    )
    lines.append(
        "    independent model state."
    )
    lines.append("")
    lines.append(
        "  • Cross-request windows use ALL C(N, B) combinations of prompts"
    )
    lines.append(
        f"    exhaustively (N={num_prompts}, B<={max(WINDOW_SIZES)}). "
        f"At B=4 that is C({num_prompts},4)={math.comb(num_prompts, 4)} "
        f"combinations; each contributes as many positions as the "
        f"shortest prompt in that combination actually generated, so "
        f"the total sample count varies by combination and is not a "
        f"single number. Prompts are NOT clamped to the global minimum. "
        "No sampling — fully"
    )
    lines.append(
        "    deterministic and reproducible under a fixed seed."
    )
    lines.append("")
    lines.append(
        "  • Consecutive condition is computed from the SAME traces in the"
    )
    lines.append(
        "    SAME run — not quoted from a previous experiment.  Both conditions"
    )
    lines.append(
        "    share identical model state and seeds."
    )
    lines.append(sep)

    report_text = "\n".join(lines)
    print(report_text)

    report_path = os.path.join(out_dir, "routing_correlation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ── Plot ───────────────────────────────────────────────────────────────────────

def _make_plot(
    results_consecutive: dict,
    results_cross: dict,
    out_dir: str,
):
    """Side-by-side heatmaps: consecutive vs cross-request per-layer ratios."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 5.5))

    for ax, (results, title) in zip(
        axes,
        [
            (results_consecutive, "Consecutive (same stream)"),
            (results_cross, "Cross-Request (independent prompts)"),
        ],
    ):
        per_layer = results["per_layer"]
        ratios = np.zeros((NUM_LAYERS, len(WINDOW_SIZES)))
        for li in range(NUM_LAYERS):
            for wi, B in enumerate(WINDOW_SIZES):
                _, _, ratio = per_layer[li][B]
                ratios[li, wi] = ratio if not np.isnan(ratio) else np.nan

        im = ax.imshow(ratios, aspect="auto", cmap="RdYlGn", vmin=0.4, vmax=1.0)
        ax.set_xticks(range(len(WINDOW_SIZES)))
        ax.set_xticklabels([f"B={B}" for B in WINDOW_SIZES])
        ax.set_yticks(range(NUM_LAYERS))
        ax.set_ylabel("Layer")
        ax.set_title(title, fontsize=11)

        for li in range(NUM_LAYERS):
            for wi in range(len(WINDOW_SIZES)):
                val = ratios[li, wi]
                color = "white" if (not np.isnan(val) and val < 0.65) else "black"
                ax.text(
                    wi, li,
                    f"{val:.2f}" if not np.isnan(val) else "N/A",
                    ha="center", va="center", fontsize=7, color=color,
                )

    fig.colorbar(im, ax=axes, label="ratio (actual/predicted)", shrink=0.8)

    # Add a second figure: B=4 ratio comparison across layers
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    layers = list(range(NUM_LAYERS))
    b4_idx = WINDOW_SIZES.index(4) if 4 in WINDOW_SIZES else 0

    b4_cons = np.array([
        results_consecutive["per_layer"][li][4][2]
        for li in range(NUM_LAYERS)
    ])
    b4_cross = np.array([
        results_cross["per_layer"][li][4][2]
        for li in range(NUM_LAYERS)
    ])

    ax2.plot(layers, b4_cons, "o-", color="steelblue", linewidth=1.5,
             markersize=5, label="Consecutive (same stream)")
    ax2.plot(layers, b4_cross, "s--", color="darkorange", linewidth=1.5,
             markersize=5, label="Cross-request (independent)")
    ax2.axhline(y=1.0, color="gray", linestyle=":", alpha=0.5,
                label="Independent (1.0)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Ratio (actual/predicted)")
    ax2.set_title("B=4 Expert Union Ratio: Consecutive vs Cross-Request")
    ax2.legend(fontsize=9)
    ax2.set_ylim(0.4, 1.05)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "routing_correlation_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    fig2.tight_layout()
    plot_path2 = os.path.join(out_dir, "routing_correlation_comparison.png")
    fig2.savefig(plot_path2, dpi=150, bbox_inches="tight")
    plt.close(fig2)

    print(f"Plots saved to {out_dir}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Cross-request routing-correlation experiment for MoE batching"
    )
    parser.add_argument(
        "--model", default=MODEL_ID,
        help=f"HF model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--num-prompts", type=int, default=8,
        help="Number of independent prompts to generate (default: 8)"
    )
    parser.add_argument(
        "--tokens-per-prompt", type=int, default=200,
        help="Tokens to generate per prompt (default: 200)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Generation temperature (default: 0.8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Base random seed (default: 42)"
    )
    parser.add_argument(
        "--out-dir", default="./cross_request_experiment_output",
        help="Output directory"
    )
    parser.add_argument(
        "--device", default=None,
        help="Device override (default: auto-detect)"
    )
    parser.add_argument(
        "--dtype", default="auto",
        choices=["auto", "float16", "bfloat16", "float32"],
        help="Model dtype (default: auto = float16 on GPU, float32 on CPU)"
    )
    parser.add_argument(
        "--prompts-file", default=None,
        help="Path to a text file with one prompt per line (overrides built-in prompts)"
    )
    parser.add_argument(
        "--force-full-length", action="store_true",
        help="Suppress EOS so every prompt generates the full token count. "
             "Without this, a prompt that reaches a natural conclusion stops "
             "early and contributes fewer positions. Note the tradeoff: text "
             "generated past a natural ending can become repetitive, and "
             "degenerate text may route differently than normal prose."
    )
    parser.add_argument(
        "--load-in-8bit", action="store_true",
        help="Load the model in 8-bit via bitsandbytes (halves memory, ~7 GB). "
             "Requires `pip install bitsandbytes accelerate`. "
             "NOTE: routing is traced on a quantized model — see caveat in output."
    )
    args = parser.parse_args()

    if args.load_in_8bit and not _HAS_BNB:
        print(
            "ERROR: --load-in-8bit requires bitsandbytes. Install with:\n"
            "  pip install bitsandbytes accelerate"
        )
        sys.exit(1)

    # ── Prompts ──────────────────────────────────────────────────────────
    if args.prompts_file:
        with open(args.prompts_file, "r") as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        prompts = DEFAULT_PROMPTS

    if args.num_prompts > len(prompts):
        print(
            f"WARNING: --num-prompts {args.num_prompts} requested but only "
            f"{len(prompts)} prompts are available; using {len(prompts)}.\n"
            f"         This matters: at B=8 with N={len(prompts)} there are "
            f"only C({len(prompts)},8)={math.comb(len(prompts), 8)} "
            f"combination(s),\n"
            f"         so the B=8 row is weakly averaged. Supply more via "
            f"--prompts-file for a trustworthy B=8 figure."
        )
    num_prompts = min(args.num_prompts, len(prompts))
    prompts = prompts[:num_prompts]

    print(f"Using {num_prompts} prompts, {args.tokens_per_prompt} tokens each")
    for i, p in enumerate(prompts):
        print(f"  [{i}] {p[:80]}...")

    # ── Device / dtype ───────────────────────────────────────────────────
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    if args.dtype == "auto":
        if device.type == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32
    elif args.dtype == "float16":
        dtype = torch.float16
    elif args.dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    quantized = args.load_in_8bit
    if quantized:
        print(f"Device: {device}, dtype: 8-bit (bitsandbytes)")
        print(
            "NOTE: routing decisions are being traced on an 8-bit-quantized "
            "model, not full precision. Router argmax is a discrete choice "
            "and mostly insulated from main-model quantization, but this is "
            "a real (second-order) deviation from measuring the model as "
            "published. Treat results as directional evidence."
        )
    else:
        print(f"Device: {device}, dtype: {dtype}")

    # ── Load model ───────────────────────────────────────────────────────
    print(f"\nLoading model {args.model} ...")
    t0 = time.time()

    if quantized:
        bnb_config = BitsAndBytesConfig(load_in_8bit=True)
        # device_map={"": 0} pins the whole model to GPU 0.  With "auto",
        # accelerate reserves headroom and may dispatch some modules to CPU,
        # which bitsandbytes 8-bit rejects outright ("Some modules are
        # dispatched on the CPU or the disk").  Pinning gives a clear OOM if
        # it genuinely does not fit, rather than a confusing config error.
        model = OlmoeForCausalLM.from_pretrained(
            args.model,
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
    else:
        model = OlmoeForCausalLM.from_pretrained(
            args.model,
            torch_dtype=dtype,
            device_map="auto" if device.type == "cuda" else None,
            attn_implementation="sdpa",
        )
        if device.type != "cuda":
            model = model.to(device)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Model loaded in {time.time() - t0:.1f}s")

    # ── Generate each prompt independently ───────────────────────────────
    print(f"\nGenerating {args.tokens_per_prompt} tokens per prompt "
          f"({num_prompts} prompts total)...")

    all_generated_ids: list[list[int]] = []
    all_expert_traces: list[dict[int, np.ndarray]] = []
    min_tokens = args.tokens_per_prompt

    for i, prompt in enumerate(prompts):
        print(f"  Prompt {i}/{num_prompts - 1}...", end=" ", flush=True)
        gen_ids, trace = generate_one_prompt(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            prompt_index=i,
            max_new_tokens=args.tokens_per_prompt,
            temperature=args.temperature,
            base_seed=args.seed,
            force_full_length=args.force_full_length,
        )
        n = len(gen_ids)
        min_tokens = min(min_tokens, n)
        all_generated_ids.append(gen_ids)
        all_expert_traces.append(trace)
        print(f"{n} tokens")

    print("\nPer-prompt generated lengths:")
    for i, tr in enumerate(all_expert_traces):
        n = next(iter(tr.values())).shape[0]
        flag = "  <-- SHORT, hit EOS early" if n < args.tokens_per_prompt else ""
        print(f"  [{i:>2}] {n:>4} tokens{flag}")

    print(f"\nMin tokens across all prompts: {min_tokens}")
    print(
        "      (analysis no longer clamps every prompt to this minimum -- each\n"
        "       prompt and each combination contributes what it actually has)"
    )
    if min_tokens < 50:
        print(
            f"WARNING: min_tokens is very low ({min_tokens}). Results will "
            "have high variance. Consider --tokens-per-prompt >= 200."
        )

    # ── Save raw data ────────────────────────────────────────────────────
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    raw_data = {
        "model": args.model,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "num_prompts": num_prompts,
        "tokens_per_prompt": args.tokens_per_prompt,
        "min_tokens_usable": min_tokens,
        "prompts": prompts,
        "generated_ids": all_generated_ids,
        "expert_traces": [
            {str(layer): trace[layer].tolist() for layer in range(NUM_LAYERS)}
            for trace in all_expert_traces
        ],
        "config": {
            "tokens_per_prompt": args.tokens_per_prompt,
            "temperature": args.temperature,
            "seed": args.seed,
            "quantized": quantized,
        },
    }
    raw_path = os.path.join(out_dir, "expert_indices.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f)
    print(f"Raw expert indices saved to {raw_path}")

    # ── Analyze ──────────────────────────────────────────────────────────
    print("\nAnalyzing consecutive condition (same-stream sliding windows)...")
    results_consecutive = analyze_consecutive(
        all_expert_traces, min_tokens
    )

    print("Analyzing cross-request condition (independent prompts)...")
    results_cross = analyze_cross_request(
        all_expert_traces, num_prompts, min_tokens
    )

    # ── Save structured results ──────────────────────────────────────────
    def _jsonify(results: dict) -> dict:
        return {
            "per_layer": {
                str(layer): {
                    str(B): {
                        "actual_mean": (
                            round(float(actual), 2)
                            if not np.isnan(actual) else None
                        ),
                        "predicted": round(float(pred), 2),
                        "ratio": (
                            round(float(ratio), 4)
                            if not np.isnan(ratio) else None
                        ),
                    }
                    for B, (actual, pred, ratio) in layer_results.items()
                }
                for layer, layer_results in results["per_layer"].items()
            },
            "overall": {
                str(B): {
                    "actual_mean": (
                        round(float(actual), 2)
                        if not np.isnan(actual) else None
                    ),
                    "predicted": round(float(pred), 2),
                    "ratio": (
                        round(float(ratio), 4)
                        if not np.isnan(ratio) else None
                    ),
                }
                for B, (actual, pred, ratio) in results["overall"].items()
            },
        }

    results_json = {
        "consecutive": _jsonify(results_consecutive),
        "cross_request": _jsonify(results_cross),
        "popularity": results_consecutive["popularity"],
        "independent_prediction": {
            str(B): {
                "predicted_union": round(predicted_union(NUM_EXPERTS, TOP_K, B), 2),
                "byte_multiplier": round(predicted_union(NUM_EXPERTS, TOP_K, B) / TOP_K, 2),
            }
            for B in WINDOW_SIZES
        },
    }
    results_path = os.path.join(out_dir, "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Analysis results saved to {results_path}")

    # ── Print report ─────────────────────────────────────────────────────
    print_report(
        results_consecutive,
        results_cross,
        num_prompts,
        args.tokens_per_prompt,
        min_tokens,
        out_dir,
        quantized=quantized,
        seed=args.seed,
    )

    # ── Plot ─────────────────────────────────────────────────────────────
    try:
        _make_plot(results_consecutive, results_cross, out_dir)
    except Exception as e:
        print(f"Plot skipped: {e}")


if __name__ == "__main__":
    main()
