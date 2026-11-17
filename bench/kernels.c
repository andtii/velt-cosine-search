// C reference for the Velt kernels: one query against every row, no top-k, best of R rounds.
// Same layout as the Velt benchmark (rows back to back). Build:
//   clang -O3 -mcpu=native bench/kernels.c -o target/kernels_c && target/kernels_c 100000 384 7
#include <arm_neon.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

static double now_ms(void) {
  struct timespec t;
  clock_gettime(CLOCK_MONOTONIC, &t);
  return t.tv_sec * 1e3 + t.tv_nsec / 1e6;
}

// Plain C loop, strict FP: clang can't reorder the sum, so it stays scalar (like Velt's dotScalar).
__attribute__((noinline)) static float dot_scalar(const float *a, const float *b, size_t d) {
  float s = 0;
  for (size_t i = 0; i < d; i++) s += a[i] * b[i];
  return s;
}

// Same loop with reassociation allowed: clang auto-vectorizes it.
#pragma clang fp reassociate(on) contract(fast)
__attribute__((noinline)) static float dot_autovec(const float *a, const float *b, size_t d) {
  float s = 0;
  for (size_t i = 0; i < d; i++) s += a[i] * b[i];
  return s;
}
#pragma clang fp reassociate(off) contract(on)

// Hand-written NEON, same shape as Velt's `dot` (4 accumulators, 16 floats per step).
__attribute__((noinline)) static float dot_neon(const float *a, const float *b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0, s2 = s0, s3 = s0;
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    s0 = vfmaq_f32(s0, vld1q_f32(a + i), vld1q_f32(b + i));
    s1 = vfmaq_f32(s1, vld1q_f32(a + i + 4), vld1q_f32(b + i + 4));
    s2 = vfmaq_f32(s2, vld1q_f32(a + i + 8), vld1q_f32(b + i + 8));
    s3 = vfmaq_f32(s3, vld1q_f32(a + i + 12), vld1q_f32(b + i + 12));
  }
  float s = vaddvq_f32(vaddq_f32(vaddq_f32(s0, s1), vaddq_f32(s2, s3)));
  for (; i < d; i++) s += a[i] * b[i];
  return s;
}

// f16 x f16 accumulated in f32 with FMLAL/FMLAL2 (FEAT_FHM), as Velt's f16x8.dotAdd.
__attribute__((noinline, target("fp16fml"))) static float dot_f16(const float16_t *a, const float16_t *b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    float16x8_t x0 = vld1q_f16(a + i), y0 = vld1q_f16(b + i);
    float16x8_t x1 = vld1q_f16(a + i + 8), y1 = vld1q_f16(b + i + 8);
    s0 = vfmlalq_low_f16(s0, x0, y0);
    s0 = vfmlalq_high_f16(s0, x0, y0);
    s1 = vfmlalq_low_f16(s1, x1, y1);
    s1 = vfmlalq_high_f16(s1, x1, y1);
  }
  float s = vaddvq_f32(vaddq_f32(s0, s1));
  for (; i < d; i++) s += (float)a[i] * (float)b[i];
  return s;
}

// bf16 with BFDOT (FEAT_BF16), as Velt's bf16x8.dotAddFast under --cpu native.
__attribute__((noinline, target("bf16"))) static float dot_bf16(const bfloat16_t *a, const bfloat16_t *b, size_t d) {
  float32x4_t s0 = vdupq_n_f32(0), s1 = s0;
  size_t i = 0;
  for (; i + 16 <= d; i += 16) {
    s0 = vbfdotq_f32(s0, vld1q_bf16(a + i), vld1q_bf16(b + i));
    s1 = vbfdotq_f32(s1, vld1q_bf16(a + i + 8), vld1q_bf16(b + i + 8));
  }
  return vaddvq_f32(vaddq_f32(s0, s1));  // d is a multiple of 16 here
}

// int8 with SDOT (FEAT_DotProd), as Velt's i32x4.dotAdd.
__attribute__((noinline, target("dotprod"))) static int32_t dot_i8(const int8_t *a, const int8_t *b, size_t d) {
  int32x4_t s0 = vdupq_n_s32(0), s1 = s0;
  size_t i = 0;
  for (; i + 32 <= d; i += 32) {
    s0 = vdotq_s32(s0, vld1q_s8(a + i), vld1q_s8(b + i));
    s1 = vdotq_s32(s1, vld1q_s8(a + i + 16), vld1q_s8(b + i + 16));
  }
  for (; i + 16 <= d; i += 16) s0 = vdotq_s32(s0, vld1q_s8(a + i), vld1q_s8(b + i));
  int32_t s = vaddvq_s32(vaddq_s32(s0, s1));
  for (; i < d; i++) s += a[i] * b[i];
  return s;
}

#define BENCH(NAME, EXPR)                                         \
  do {                                                            \
    double best = 1e30;                                           \
    for (int r = 0; r < rounds; r++) {                            \
      double t0 = now_ms();                                       \
      double s = 0;                                               \
      for (size_t row = 0; row < n; row++) s += (EXPR);           \
      double el = now_ms() - t0;                                  \
      sink += s;                                                  \
      if (el < best) best = el;                                   \
    }                                                             \
    printf("| C %s | %.2f |\n", NAME, best * 1e6 / (double)n);    \
  } while (0)

int main(int argc, char **argv) {
  size_t n = argc > 1 ? strtoul(argv[1], 0, 10) : 100000;
  size_t d = argc > 2 ? strtoul(argv[2], 0, 10) : 384;
  int rounds = argc > 3 ? atoi(argv[3]) : 7;
  float *f = malloc(n * d * sizeof(float));
  float16_t *h = malloc(n * d * sizeof(float16_t));
  bfloat16_t *bf = malloc(n * d * sizeof(bfloat16_t));
  int8_t *q8 = malloc(n * d);
  for (size_t i = 0; i < n * d; i++) {
    float x = (float)rand() / RAND_MAX * 2 - 1;
    f[i] = x;
    h[i] = (float16_t)x;
    bf[i] = vcvth_bf16_f32(x);
    q8[i] = (int8_t)(x * 127);
  }
  double sink = 0;
  printf("| C kernel (x1, no top-k), n=%zu dim=%zu | ns/vector (best) |\n|---|---|\n", n, d);
  BENCH("f32 scalar (strict FP)", dot_scalar(f + row * d, f, d));
  BENCH("f32 auto-vectorized (reassoc)", dot_autovec(f + row * d, f, d));
  BENCH("f32 NEON 4-acc FMA", dot_neon(f + row * d, f, d));
  BENCH("f16 FMLAL", dot_f16(h + row * d, h, d));
  BENCH("bf16 BFDOT", dot_bf16(bf + row * d, bf, d));
  BENCH("i8 SDOT", dot_i8(q8 + row * d, q8, d));
  fprintf(stderr, "sink %g\n", sink);
  return 0;
}
