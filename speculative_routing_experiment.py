#!/usr/bin/env python3
"""
Speculative decoding routing-correlation experiment for a streaming MoE.

Question: When OLMoE-1B-7B generates consecutive tokens, how much do the routed
experts overlap across those tokens?

This determines whether speculative decoding helps or hurts on a
storage-bandwidth-bound MoE inference system.

If routing is highly correlated (actual union << predicted union under independent
sampling), speculative decoding pays: verifying B draft tokens loads far fewer distinct
experts than B independent tokens would.

If routing is near-independent (actual union ≈ predicted), speculative decoding is a net
loss: you pay for the union of experts but only get ~2.2 tokens from 4 drafts.

Model: OLMoE-1B-7B (16 layers, 64 experts, top-8 routing, dropless).
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, OlmoeForCausalLM

# ── Constants ──────────────────────────────────────────────────────────────────
MODEL_ID = "allenai/OLMoE-1B-7B-0924"
NUM_EXPERTS = 64
TOP_K = 8
NUM_LAYERS = 16
WINDOW_SIZES = [2, 4, 8]

# A realistic prompt mixing prose and code, so the model generates a mix of both.
# Two Wikipedia-style paragraphs + a Python function.
PROMPT_TEXT = """\
The Transformer architecture, introduced by Vaswani et al. in 2017, replaced recurrent
neural networks with a purely attention-based mechanism for sequence transduction tasks.
Its key innovation is the self-attention layer, which computes a weighted sum of all
input positions in parallel, rather than processing tokens sequentially. This allows the
model to capture long-range dependencies more effectively and enables massive
parallelism during training. The original Transformer consists of an encoder and a
decoder, each built from stacked layers of multi-head attention and position-wise
feed-forward networks. Residual connections and layer normalization stabilize training,
while sinusoidal positional encodings inject order information into the otherwise
permutation-invariant attention mechanism. Since its publication, the Transformer has
become the foundation of virtually all state-of-the-art natural language processing
systems, including BERT, GPT, T5, and their numerous descendants.

The Mixture-of-Experts (MoE) paradigm extends the Transformer by replacing the dense
feed-forward layers with a set of specialized "expert" sub-networks, each trained on
different aspects of the data. A learned router selects a subset of experts for each
input token, typically using a top-k gating mechanism with a softmax over expert
affinities followed by a load-balancing auxiliary loss. The key advantage is
computational: an MoE model can scale to hundreds of billions of total parameters while
keeping the per-token compute cost proportional only to the active experts, not the full
model. This sparsity makes MoE models dramatically more parameter-efficient than their
dense counterparts. However, it also introduces challenges: expert load imbalance can
cause some experts to be overused while others sit idle, the discrete routing decision
is non-differentiable and requires careful gradient estimation, and at inference time
the active expert weights must be fetched from memory, making storage bandwidth the
binding constraint for batch-1 decode on large MoE models.

def fibonacci_search(arr: list[int], target: int) -> int:
    \"\"\"Search a sorted array using Fibonacci numbers for probe positions.
    
    Fibonacci search is a comparison-based technique that uses Fibonacci
    numbers to partition the search space, avoiding the division operation
    required by binary search. It can be faster on systems where addition
    and subtraction are cheaper than division.
    
    Args:
        arr: A sorted list of integers to search within.
        target: The integer value to locate.
    
    Returns:
        The index of target in arr, or -1 if not found.
    \"\"\"
    n = len(arr)
    if n == 0:
        return -1
    
    # Initialize Fibonacci numbers
    fib_m2 = 0  # F(k-2)
    fib_m1 = 1  # F(k-1)
    fib_m = fib_m2 + fib_m1  # F(k)
    
    # Find smallest Fibonacci >= n
    while fib_m < n:
        fib_m2 = fib_m1
        fib_m1 = fib_m
        fib_m = fib_m2 + fib_m1
    
    offset = -1
    while fib_m > 1:
        i = min(offset + fib_m2, n - 1)
        if arr[i] < target:
            fib_m = fib_m1
            fib_m1 = fib_m2
            fib_m2 = fib_m - fib_m1
            offset = i
        elif arr[i] > target:
            fib_m = fib_m2
            fib_m1 = fib_m1 - fib_m2
            fib_m2 = fib_m - fib_m1
        else:
            return i
    
    if fib_m1 and offset + 1 < n and arr[offset + 1] == target:
        return offset + 1
    return -1


# The following text continues the discussion of efficient algorithms and
# their practical applications in modern computing systems.
"""


# ── Hook machinery ─────────────────────────────────────────────────────────────
def install_router_hooks(model: OlmoeForCausalLM) -> dict:
    """Register forward hooks on every OlmoeTopKRouter (gate) in the model.

    Returns a dict to be populated: captured[layer_idx].append(selected_experts_array)
    The gate returns (router_logits, router_scores, router_indices); we capture indices.
    """
    captured = defaultdict(list)

    def make_hook(layer_idx: int):
        def hook(module, input_, output):
            # OlmoeTopKRouter returns (router_logits, router_scores, router_indices)
            # router_indices: (batch*seq_len, top_k) LongTensor of expert indices
            selected_experts = output[2].detach().cpu().numpy().astype(np.int16)
            captured[layer_idx].append(selected_experts)

        return hook

    for i in range(NUM_LAYERS):
        gate = model.model.layers[i].mlp.gate
        gate.register_forward_hook(make_hook(i))

    return captured


# ── Independent-routing prediction ─────────────────────────────────────────────
def predicted_union(e: int, k: int, b: int) -> float:
    """Expected distinct experts when B tokens independently select k from e."""
    return e * (1.0 - (1.0 - k / e) ** b)


# ── Generation ─────────────────────────────────────────────────────────────────
def generate_with_routing_trace(
    model: OlmoeForCausalLM,
    tokenizer: AutoTokenizer,
    prompt: str,
    max_new_tokens: int = 1500,
    temperature: float = 0.8,
    top_p: float = 0.95,
    seed: int = 42,
) -> tuple[list[int], dict[int, np.ndarray]]:
    """Generate text and capture expert selections for every generated token.

    Returns:
        generated_ids: token IDs of the generated portion (excluding prompt).
        expert_trace: dict mapping layer_idx -> array of shape (n_gen_tokens, top_k)
                      with expert indices.
    """
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
            top_p=top_p,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0, prompt_len:].tolist()
    n_gen = len(generated_ids)

    print(f"Generated {n_gen} tokens in {elapsed:.1f}s ({n_gen / elapsed:.1f} tok/s)")

    # Concatenate captured chunks per layer.
    # During prefill the hook fires once with (prompt_len, top_k).
    # During generation it fires once per token with (1, top_k).
    # We only want generated tokens, so we slice off prompt_len from the
    # first chunk (prefill) and concatenate all subsequent chunks.
    expert_trace = {}
    for layer_idx in range(NUM_LAYERS):
        chunks = captured[layer_idx]
        if not chunks:
            raise RuntimeError(f"No router outputs captured for layer {layer_idx}")

        # First chunk contains prompt + possibly some generated tokens.
        # Actually, generate() does prefill in one forward pass (prompt_len tokens),
        # then N generation steps of 1 token each.  The first hook fires on prefill
        # with (prompt_len, top_k); subsequent hooks fire on individual generation
        # steps with (1, top_k).
        gen_chunks = []
        for i, ch in enumerate(chunks):
            if i == 0:
                # First chunk is the prefill; discard prompt portion.
                if ch.shape[0] > prompt_len:
                    # Some generated tokens might be in here? Unlikely but handle it.
                    gen_chunks.append(ch[prompt_len:])
                # else: all prompt, skip
            else:
                gen_chunks.append(ch)

        if not gen_chunks:
            raise RuntimeError(
                f"No generated-token router outputs for layer {layer_idx}. "
                f"Did generation produce any tokens?"
            )

        trace = np.concatenate(gen_chunks, axis=0)
        if trace.shape[0] != n_gen:
            print(
                f"  Warning: layer {layer_idx} trace has {trace.shape[0]} rows, "
                f"expected {n_gen}. Truncating/padding."
            )
            if trace.shape[0] > n_gen:
                trace = trace[:n_gen]
            else:
                pad = np.full((n_gen - trace.shape[0], TOP_K), -1, dtype=np.int16)
                trace = np.concatenate([trace, pad], axis=0)

        expert_trace[layer_idx] = trace

    return generated_ids, expert_trace


# ── Analysis ───────────────────────────────────────────────────────────────────
def analyze_routing_correlation(
    expert_trace: dict[int, np.ndarray],
) -> dict:
    """Compute actual vs predicted expert unions for sliding windows.

    Returns a dict with:
        per_layer: dict[layer_idx][window_size] = (actual_mean, predicted, ratio)
        overall: dict[window_size] = (actual_mean, predicted, ratio)  averaged across layers
        popularity: dict with top_5pct_fraction and per_expert_counts
    """
    n_tokens = next(iter(expert_trace.values())).shape[0]
    print(f"\nAnalyzing {n_tokens} generated tokens across {NUM_LAYERS} layers...")

    results_per_layer = {}
    expert_counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)

    for layer_idx in range(NUM_LAYERS):
        trace = expert_trace[layer_idx]  # (n_tokens, top_k)
        layer_results = {}

        for B in WINDOW_SIZES:
            if n_tokens < B:
                layer_results[B] = (np.nan, predicted_union(NUM_EXPERTS, TOP_K, B), np.nan)
                continue

            actual_unions = []
            for start in range(n_tokens - B + 1):
                window = trace[start : start + B]  # (B, top_k)
                union_size = len(set(window.flatten().tolist()))
                actual_unions.append(union_size)

            actual_mean = np.mean(actual_unions)
            predicted = predicted_union(NUM_EXPERTS, TOP_K, B)
            ratio = actual_mean / predicted
            layer_results[B] = (actual_mean, predicted, ratio)

        results_per_layer[layer_idx] = layer_results

        # Tally expert selections for popularity analysis
        for t in range(n_tokens):
            for e in trace[t]:
                expert_counts[layer_idx, e] += 1

    # Overall averages
    overall = {}
    for B in WINDOW_SIZES:
        actuals = []
        for layer_idx in range(NUM_LAYERS):
            a, _, _ = results_per_layer[layer_idx][B]
            if not np.isnan(a):
                actuals.append(a)
        pred = predicted_union(NUM_EXPERTS, TOP_K, B)
        overall[B] = (np.mean(actuals), pred, np.mean(actuals) / pred)

    # Expert popularity: top-5% most selected experts and their share
    popularity = {}
    for layer_idx in range(NUM_LAYERS):
        counts = expert_counts[layer_idx]
        total_selections = counts.sum()
        sorted_desc = np.sort(counts)[::-1]
        top_5pct_n = max(1, int(np.ceil(NUM_EXPERTS * 0.05)))  # top 5% of 64 = 4
        top_5pct_count = sorted_desc[:top_5pct_n].sum()
        top_5pct_fraction = top_5pct_count / total_selections if total_selections > 0 else 0
        popularity[layer_idx] = {
            "top_5pct_fraction": round(top_5pct_fraction, 4),
            "top_5pct_experts": [
                int(e) for e in np.argsort(counts)[::-1][:top_5pct_n]
            ],
            "expert_counts": counts.tolist(),
        }

    return {
        "per_layer": results_per_layer,
        "overall": overall,
        "popularity": popularity,
    }


# ── Output ─────────────────────────────────────────────────────────────────────
def print_report(results: dict, generated_ids: list[int], out_dir: str):
    """Print formatted results and write the interpretation paragraph."""
    per_layer = results["per_layer"]
    overall = results["overall"]
    popularity = results["popularity"]

    n_tokens = len(generated_ids)
    pred_ref = {B: predicted_union(NUM_EXPERTS, TOP_K, B) for B in WINDOW_SIZES}

    # ── Table ──────────────────────────────────────────────────────────────
    header = f"{'Layer':>6}"
    for B in WINDOW_SIZES:
        header += f"  B={B} actual  pred  ratio"
    sep = "-" * len(header)

    lines = [sep, header, sep]
    for layer_idx in range(NUM_LAYERS):
        row = f"{layer_idx:>6}"
        for B in WINDOW_SIZES:
            actual, pred, ratio = per_layer[layer_idx][B]
            if np.isnan(actual):
                row += f"  {'':>6}  {pred:5.1f}  {'':>5}"
            else:
                row += f"  {actual:6.2f}  {pred:5.1f}  {ratio:.3f}"
        lines.append(row)
    lines.append(sep)
    # Overall row
    row = "OVERALL"
    for B in WINDOW_SIZES:
        actual, pred, ratio = overall[B]
        row += f"  {actual:6.2f}  {pred:5.1f}  {ratio:.3f}"
    lines.append(row)
    lines.append(sep)

    # ── Interpretation ─────────────────────────────────────────────────────
    # Break-even is derived from architecture.md section 6.5, not an
    # imported round number. The independent-sampling prediction already
    # encodes "B tokens, no correlation" as a *bytes* multiplier (union
    # size relative to a single token's k=8 experts). What decides whether
    # speculation pays is whether that bytes multiplier is smaller than the
    # tokens you actually get accepted for paying it.
    #
    #   bytes_multiplier(B) = predicted_union(B) / k        (independent case)
    #   measured_multiplier(B) = actual_union_mean(B) / k    (from this trace)
    #   break_even_multiplier(B) = B / accepted(B)
    #
    # Speculation pays iff measured_multiplier(B) < break_even_multiplier(B).
    #
    # accepted(B) is NOT measured by this script -- it depends on the draft
    # model's quality, which hasn't been chosen yet. The 2.2-at-B=4 figure
    # in architecture.md is an assumption carried over from typical EAGLE/
    # MTP acceptance rates on dense models, not something specific to this
    # MoE or measured here. We report the verdict *conditional on that
    # assumption* and print the assumption explicitly so it can't be missed.

    ASSUMED_ACCEPTANCE = {2: 1.6, 4: 2.2, 8: 3.4}  # tokens accepted, per B; placeholder, see note above

    def multiplier(actual_union_mean: float) -> float:
        return actual_union_mean / TOP_K

    print_rows = []
    verdicts = {}
    for B in WINDOW_SIZES:
        actual, pred, ratio = overall[B]
        accepted = ASSUMED_ACCEPTANCE.get(B)
        measured_mult = multiplier(actual)
        indep_mult = multiplier(pred)
        break_even_mult = B / accepted if accepted else float("nan")
        pays = measured_mult < break_even_mult
        verdicts[B] = pays
        print_rows.append(
            f"  B={B}: measured {measured_mult:.2f}x bytes vs break-even "
            f"{break_even_mult:.2f}x (needs < break-even to pay) "
            f"-> {'PAYS' if pays else 'NET LOSS'}  "
            f"[independent-routing case would be {indep_mult:.2f}x]"
        )

    b4_pays = verdicts.get(4)
    if b4_pays is True:
        guidance = (
            "ENABLE speculative decoding. Under the assumed acceptance rate "
            "of ~2.2 tokens at B=4, measured routing correlation is high "
            "enough that verifying 4 draft tokens loads fewer bytes than "
            "the break-even point. This conclusion depends on the assumed "
            "acceptance rate above -- re-check if the actual draft model's "
            "acceptance differs materially from 2.2/4."
        )
    elif b4_pays is False:
        guidance = (
            "DISABLE speculative decoding on this streaming architecture. "
            "Even crediting the assumed ~2.2 tokens accepted at B=4, the "
            "measured expert union is large enough that verification costs "
            "more bytes than the break-even point allows. This holds unless "
            "the real draft model's acceptance rate is substantially higher "
            "than assumed."
        )
    else:
        guidance = "Could not compute a verdict -- insufficient tokens generated."

    interp = (
        f"OLMoE-1B-7B generated {n_tokens} tokens.\n\n"
        f"Break-even analysis (architecture.md section 6.5), assuming "
        f"acceptance rates {ASSUMED_ACCEPTANCE} -- THESE ARE PLACEHOLDERS, "
        f"not measured, since no draft model was run here:\n"
        + "\n".join(print_rows)
        + f"\n\n{guidance}"
    )

    # ── Popularity summary ─────────────────────────────────────────────────
    pop_lines = []
    avg_top5_frac = np.mean(
        [popularity[l]["top_5pct_fraction"] for l in range(NUM_LAYERS)]
    )
    pop_lines.append(
        f"\nExpert popularity: top-5% ({max(1, int(np.ceil(NUM_EXPERTS * 0.05)))} of "
        f"{NUM_EXPERTS}) experts capture on average {avg_top5_frac:.1%} of all "
        f"routing decisions across layers. This feeds the expert-caching question."
    )

    # ── Print and save ─────────────────────────────────────────────────────
    report_text = "\n".join(lines + [""] + pop_lines + ["", interp])
    print(report_text)

    report_path = os.path.join(out_dir, "routing_correlation_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"\nReport saved to {report_path}")


# ── Main ───────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(
        description="Speculative decoding routing-correlation experiment for MoE"
    )
    parser.add_argument(
        "--model", default=MODEL_ID,
        help=f"HF model ID (default: {MODEL_ID})"
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=1500,
        help="Number of tokens to generate (default: 1500)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.8,
        help="Generation temperature (default: 0.8)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    parser.add_argument(
        "--out-dir", default="./routing_experiment_output",
        help="Output directory (default: ./routing_experiment_output)"
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
        "--prompt-file", default=None,
        help="Path to a text file to use as prompt (overrides built-in prompt)"
    )
    args = parser.parse_args()

    # ── Device / dtype ─────────────────────────────────────────────────────
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

    print(f"Device: {device}, dtype: {dtype}")

    # ── Load model ─────────────────────────────────────────────────────────
    print(f"Loading model {args.model} ...")
    t0 = time.time()

    # Use SDPA attention (recommended by the model card)
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

    # ── Prompt ─────────────────────────────────────────────────────────────
    if args.prompt_file:
        with open(args.prompt_file, "r") as f:
            prompt = f.read()
    else:
        prompt = PROMPT_TEXT

    # ── Generate with trace capture ────────────────────────────────────────
    generated_ids, expert_trace = generate_with_routing_trace(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        seed=args.seed,
    )

    # ── Save raw data ──────────────────────────────────────────────────────
    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)

    # Save expert indices (convert numpy to list for JSON)
    raw_data = {
        "model": args.model,
        "num_layers": NUM_LAYERS,
        "num_experts": NUM_EXPERTS,
        "top_k": TOP_K,
        "num_generated_tokens": len(generated_ids),
        "generated_ids": generated_ids,
        "expert_trace": {
            str(layer): expert_trace[layer].tolist()
            for layer in range(NUM_LAYERS)
        },
        "config": {
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "seed": args.seed,
        },
    }
    raw_path = os.path.join(out_dir, "expert_indices.json")
    with open(raw_path, "w") as f:
        json.dump(raw_data, f)
    print(f"Raw expert indices saved to {raw_path}")

    # ── Analyze ────────────────────────────────────────────────────────────
    results = analyze_routing_correlation(expert_trace)

    # Save structured results as JSON
    results_json = {
        "per_layer": {
            str(layer): {
                str(B): {
                    "actual_mean": round(float(actual), 2),
                    "predicted": round(float(pred), 2),
                    "ratio": round(float(ratio), 4),
                }
                for B, (actual, pred, ratio) in layer_results.items()
            }
            for layer, layer_results in results["per_layer"].items()
        },
        "overall": {
            str(B): {
                "actual_mean": round(float(actual), 2),
                "predicted": round(float(pred), 2),
                "ratio": round(float(ratio), 4),
            }
            for B, (actual, pred, ratio) in results["overall"].items()
        },
        "popularity": results["popularity"],
    }
    results_path = os.path.join(out_dir, "analysis_results.json")
    with open(results_path, "w") as f:
        json.dump(results_json, f, indent=2)
    print(f"Analysis results saved to {results_path}")

    # ── Print report ───────────────────────────────────────────────────────
    print_report(results, generated_ids, out_dir)

    # ── Optional plot ──────────────────────────────────────────────────────
    try:
        _make_plot(results, out_dir)
    except Exception as e:
        print(f"Plot skipped: {e}")


def _make_plot(results: dict, out_dir: str):
    """Generate a plot: layer × window size heatmap of actual/predicted ratio."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    per_layer = results["per_layer"]

    layers = list(range(NUM_LAYERS))
    ratios = np.zeros((NUM_LAYERS, len(WINDOW_SIZES)))
    for li, layer_idx in enumerate(layers):
        for wi, B in enumerate(WINDOW_SIZES):
            _, _, ratio = per_layer[layer_idx][B]
            ratios[li, wi] = ratio if not np.isnan(ratio) else np.nan

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Heatmap: layer × window size
    im = ax1.imshow(ratios, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=1.0)
    ax1.set_xticks(range(len(WINDOW_SIZES)))
    ax1.set_xticklabels([f"B={B}" for B in WINDOW_SIZES])
    ax1.set_yticks(range(NUM_LAYERS))
    ax1.set_ylabel("Layer")
    ax1.set_title("Actual / Predicted Expert Union Ratio")
    plt.colorbar(im, ax=ax1, label="ratio")

    # Annotate
    for li in range(NUM_LAYERS):
        for wi in range(len(WINDOW_SIZES)):
            val = ratios[li, wi]
            color = "white" if (not np.isnan(val) and val < 0.6) else "black"
            ax1.text(wi, li, f"{val:.2f}" if not np.isnan(val) else "N/A",
                     ha="center", va="center", fontsize=7, color=color)

    # Per-layer ratio line plot (B=4)
    b4_idx = WINDOW_SIZES.index(4) if 4 in WINDOW_SIZES else 0
    b4_ratios = ratios[:, b4_idx]
    ax2.plot(layers, b4_ratios, "o-", color="steelblue", linewidth=1.5, markersize=4)
    ax2.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Independent (1.0)")
    ax2.axhline(y=0.5, color="red", linestyle=":", alpha=0.5, label="Strong correlation (0.5)")
    ax2.set_xlabel("Layer")
    ax2.set_ylabel("Ratio (actual/predicted)")
    ax2.set_title("B=4 Expert Union Ratio by Layer")
    ax2.legend(fontsize=8)
    ax2.set_ylim(0, 1.1)

    fig.tight_layout()
    plot_path = os.path.join(out_dir, "routing_correlation_plot.png")
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Plot saved to {plot_path}")


if __name__ == "__main__":
    main()
