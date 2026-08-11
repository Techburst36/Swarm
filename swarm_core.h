#ifndef SWARM_CORE_H
#define SWARM_CORE_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

/* ── F32 matrix-vector multiply ───────────────────────────────────────────
 * output = weights @ input
 *
 * weights: [rows, cols] row-major float32
 * input:   [cols] float32
 * output:  [rows] float32 (caller-allocated)
 *
 * Returns 0 on success, -1 on invalid arguments.
 */
int gemv_f32(const float *weights, const float *input, float *output,
             int rows, int cols);

/* ── Q4_K_M quantized matrix-vector multiply ─────────────────────────────
 * Dequantizes and multiplies in one fused pass.
 *
 * Q4_K_M block layout (llama.cpp / ggml compatible):
 *   ┌──────────┬──────────┬───────────────┬───────────────┐
 *   │ d  (f16) │ dmin(f16)│ scales (12 B) │  qs  (128 B)  │
 *   │   2 B    │   2 B    │ 16×6-bit sc   │ 256×4-bit q   │
 *   └──────────┴──────────┴───────────────┴───────────────┘
 *   Total: 144 bytes per 256-element block
 *
 * Dequant:  val = d * sc * (q - 8) - dmin
 *
 * weights: rows * ceil(cols / 256) * 144  bytes
 * input:   [cols] float32
 * output:  [rows] float32 (caller-allocated)
 *
 * Non-256-aligned edge columns handled via scalar fallback.
 */
int gemv_q4km(const void *weights, const float *input, float *output,
              int rows, int cols);

/* ── Utility: AVX2-capable? ────────────────────────────────────────────── */
int swarm_core_has_avx2(void);

/* ── Version ───────────────────────────────────────────────────────────── */
const char *swarm_core_version(void);

#ifdef __cplusplus
}
#endif
#endif /* SWARM_CORE_H */
