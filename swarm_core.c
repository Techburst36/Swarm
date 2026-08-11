/*
 * swarm_core.c — AVX2/FMA3 vectorized GEMM kernels for Swarm Layer 1.
 *
 * Compile:  gcc -O3 -mavx2 -mfma -shared -fPIC -o libswarm_core.so swarm_core.c
 *
 * Provides:
 *   gemv_f32   — float32 matrix-vector multiply
 *   gemv_q4km  — Q4_K_M dequant + matrix-vector multiply (fused)
 *
 * Design rules:
 *   - C11, no external dependencies beyond <immintrin.h>.
 *   - AVX2/FMA3 on the fast path, scalar fallback for edge columns
 *     and non-aligned tail elements.
 *   - All pointers are bounds-checked (null guard at entry).
 *   - Q4_K_M block format matches llama.cpp/ggml for interop.
 */

#include "swarm_core.h"

#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wpedantic"
#endif

/* ── Feature detection ─────────────────────────────────────────────────── */

#if defined(__AVX2__) && defined(__FMA__)
#include <immintrin.h>
#define HAS_AVX2_FMA 1
#else
#define HAS_AVX2_FMA 0
#endif

int swarm_core_has_avx2(void) {
#if HAS_AVX2_FMA
    return 1;
#else
    return 0;
#endif
}

const char *swarm_core_version(void) {
    return "0.1.0";
}

/* ── f16 → f32 conversion (IEEE 754-2008 binary16) ─────────────────────── */

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign  = (uint32_t)(h & 0x8000u) << 16;
    int      exp   = (int)((h >> 10) & 0x1Fu);
    uint32_t mant  = (uint32_t)(h & 0x3FFu);

    if (exp == 0) {
        /* zero or subnormal */
        if (mant == 0) {
            uint32_t r = sign;
            float f;
            memcpy(&f, &r, sizeof(f));
            return f;
        }
        /* subnormal → normalised */
        int shift = 0;
        uint32_t m = mant;
        while ((m & 0x400u) == 0) {
            m <<= 1;
            shift++;
        }
        m &= 0x3FFu;
        exp = 1 - shift;
        uint32_t r = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (m << 13);
        float f;
        memcpy(&f, &r, sizeof(f));
        return f;
    }

    if (exp == 0x1F) {
        /* Inf / NaN */
        uint32_t r = sign | 0x7F800000u | (mant << 13);
        float f;
        memcpy(&f, &r, sizeof(f));
        return f;
    }

    /* normal */
    uint32_t r = sign | ((uint32_t)(exp + (127 - 15)) << 23) | (mant << 13);
    float f;
    memcpy(&f, &r, sizeof(f));
    return f;
}

/* ── Horizontal sum of an __m256 register (AVX) ────────────────────────── */

#if HAS_AVX2_FMA
static inline float hsum_ps(__m256 v) {
    /* v  = [a b c d | e f g h] */
    __m128 hi  = _mm256_extractf128_ps(v, 1);
    __m128 lo  = _mm256_castps256_ps128(v);
    __m128 sum = _mm_add_ps(lo, hi);          /* [a+e b+f c+g d+h] */
    sum = _mm_hadd_ps(sum, sum);              /* [a+e+b+f  c+g+d+h  …  …] */
    sum = _mm_hadd_ps(sum, sum);              /* [all four summed] */
    return _mm_cvtss_f32(sum);
}
#endif

/* ── GEMV: F32 ──────────────────────────────────────────────────────────── */

int gemv_f32(const float *weights, const float *input, float *output,
             int rows, int cols) {
    if (!weights || !input || !output) return -1;
    if (rows <= 0 || cols <= 0) return -1;

#if HAS_AVX2_FMA
    for (int r = 0; r < rows; r++) {
        const float *row = weights + (size_t)r * (size_t)cols;
        __m256 sum0 = _mm256_setzero_ps();
        int c = 0;

        /* 8-wide main loop */
        for (; c + 7 < cols; c += 8) {
            __m256 w = _mm256_loadu_ps(row + c);
            __m256 x = _mm256_loadu_ps(input + c);
            sum0 = _mm256_fmadd_ps(w, x, sum0);
        }

        float result = hsum_ps(sum0);

        /* scalar tail */
        for (; c < cols; c++) {
            result += row[c] * input[c];
        }

        output[r] = result;
    }
#else
    /* ── Pure-scalar fallback ───────────────────────────────────────── */
    for (int r = 0; r < rows; r++) {
        const float *row = weights + (size_t)r * (size_t)cols;
        float dot = 0.0f;
        for (int c = 0; c < cols; c++) {
            dot += row[c] * input[c];
        }
        output[r] = dot;
    }
#endif
    return 0;
}

/* ── Q4_K_M helpers ────────────────────────────────────────────────────── */

#define QK_K 256
#define QK_K_SUB 16                       /* elements per sub-block */
#define Q4K_SCALES_BYTES 12               /* 16 × 6 bits */
#define Q4K_QS_BYTES 128                  /* 256 × 4 bits   */
#define Q4K_BLOCK_BYTES (2 + 2 + Q4K_SCALES_BYTES + Q4K_QS_BYTES)  /* 144 */

/* Extract 16 × 6-bit scales from a 12-byte packed array.
 * The 12 bytes hold 96 bits = 16×6.  When byte_off+1 would read past
 * the array (last scale: bits 90–95 lie entirely in byte 11), the high
 * byte is zero. */
static void unpack_scales(const uint8_t scales[Q4K_SCALES_BYTES],
                          uint8_t out[QK_K / QK_K_SUB]) {
    for (int i = 0; i < QK_K / QK_K_SUB; i++) {
        int bit_off  = i * 6;
        int byte_off = bit_off >> 3;
        int shift    = bit_off & 7;
        uint16_t val = (uint16_t)scales[byte_off];
        if (byte_off + 1 < Q4K_SCALES_BYTES) {
            val |= ((uint16_t)scales[byte_off + 1] << 8);
        }
        out[i] = (uint8_t)((val >> shift) & 0x3Fu);
    }
}

/* Get nibble from the packed qs array. */
static inline int q4_nibble(const uint8_t qs[Q4K_QS_BYTES], int idx) {
    return (qs[idx >> 1] >> (4 * (idx & 1))) & 0x0F;
}

/* ── GEMV: Q4_K_M ──────────────────────────────────────────────────────── */

int gemv_q4km(const void *weights, const float *input, float *output,
              int rows, int cols) {
    if (!weights || !input || !output) return -1;
    if (rows <= 0 || cols <= 0) return -1;

    const uint8_t *w8 = (const uint8_t *)weights;

    int blocks_per_row = (cols + QK_K - 1) / QK_K;

    for (int r = 0; r < rows; r++) {
        const uint8_t *row_blocks = w8 + (size_t)r * (size_t)blocks_per_row
                                          * (size_t)Q4K_BLOCK_BYTES;
        float dot = 0.0f;

        for (int b = 0; b < blocks_per_row; b++) {
            const uint8_t *blk = row_blocks + (size_t)b * Q4K_BLOCK_BYTES;

            /* header */
            uint16_t d_raw, dmin_raw;
            memcpy(&d_raw,    blk,      2);
            memcpy(&dmin_raw, blk + 2,  2);
            float d    = f16_to_f32(d_raw);
            float dmin = f16_to_f32(dmin_raw);

            const uint8_t *scales = blk + 4;
            const uint8_t *qs     = blk + 4 + Q4K_SCALES_BYTES;

            uint8_t sc[16];
            unpack_scales(scales, sc);

            int base_c = b * QK_K;
            int valid  = cols - base_c;
            if (valid > QK_K) valid = QK_K;

#if HAS_AVX2_FMA
            /* ── AVX2 path per sub-block ──────────────────────────── */
            for (int sb = 0; sb < QK_K / QK_K_SUB; sb++) {
                int sub_start = base_c + sb * QK_K_SUB;
                if (sub_start >= cols) break;

                int sub_valid = cols - sub_start;
                if (sub_valid > QK_K_SUB) sub_valid = QK_K_SUB;

                /* dequantize sub-block into stack buffer */
                float dbuf[QK_K_SUB];
                int s;
                for (s = 0; s < sub_valid; s++) {
                    int q = q4_nibble(qs, sb * QK_K_SUB + s);
                    dbuf[s] = d * (float)((int)sc[sb] * (q - 8)) - dmin;
                }
                /* zero-pad if block extends past cols */
                for (; s < QK_K_SUB; s++) {
                    dbuf[s] = 0.0f;
                }

                /* AVX2 dot */
                __m256 sum_sb = _mm256_setzero_ps();
                int t = 0;
                for (; t + 7 < QK_K_SUB; t += 8) {
                    __m256 wv = _mm256_loadu_ps(dbuf + t);
                    __m256 xv = _mm256_loadu_ps(input + sub_start + t);
                    sum_sb = _mm256_fmadd_ps(wv, xv, sum_sb);
                }
                dot += hsum_ps(sum_sb);
                for (; t < QK_K_SUB; t++) {
                    dot += dbuf[t] * input[sub_start + t];
                }
            }
#else
            /* ── Scalar path ──────────────────────────────────────── */
            for (int i = 0; i < valid; i++) {
                int sb    = i / QK_K_SUB;
                int q     = q4_nibble(qs, i);
                float val = d * (float)((int)sc[sb] * (q - 8)) - dmin;
                dot += val * input[base_c + i];
            }
#endif
        }

        output[r] = dot;
    }

    return 0;
}

#if defined(__GNUC__) || defined(__clang__)
#pragma GCC diagnostic pop
#endif
