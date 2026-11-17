# Velt 0.6 SIMD vector search: findings

A brute-force cosine similarity search written in [Velt](https://veltlang.com) 0.6.0, with vectors stored as f32, f16, bf16 or int8. It was built to test Velt's SIMD and numeric support on a real workload and report what helped and what got in the way. Code, benchmarks and repros are all in this repo.

**Result.** On an Apple M4, Velt's f32 SIMD kernel runs within 1–6% of hand-written C NEON. int8 search of 100k × 384 vectors takes about 1 ms on 10 cores at 0.975–0.99 recall@10. Four problems cost the most:

1. Closures that deep-copied their captures on every call (21× slower end to end). **Already fixed on main** by [velt-lang/velt#2734](https://github.com/velt-lang/velt/pull/2734), after the 0.6.0 release.
2. `bf16x8.dotAdd` is 3.7× slower than C's `bfdot`, because it quiets NaN payloads lane by lane. Filed as [velt-lang/velt#3129](https://github.com/velt-lang/velt/issues/3129).
3. Bounds checks that only go away with the `fits()` idiom, and an array pointer reloaded every iteration. Partly tracked.
4. No `exp`, `log` or trig functions in `Math`. Filed as [velt-lang/velt#3130](https://github.com/velt-lang/velt/issues/3130).

## Contents

- [Summary of findings](#summary-of-findings)
- [Existing issues and PRs](#existing-issues-and-prs)
- [Setup](#setup)
- [Results by storage format](#results-by-storage-format)
- [Kernels against C NEON](#kernels-against-c-neon)
- [Finding 1: closures copied their captures on every call (fixed on main)](#finding-1-closures-copied-their-captures-on-every-call-fixed-on-main)
- [Finding 2: bf16x8.dotAdd is 3–4× slower than f16](#finding-2-bf16x8dotadd-is-34-slower-than-f16)
- [Finding 3: bounds checks and pointer reloads in SIMD loops](#finding-3-bounds-checks-and-pointer-reloads-in-simd-loops)
- [Finding 4: Math has no exp, log or trig functions](#finding-4-math-has-no-exp-log-or-trig-functions)
- [Smaller findings](#smaller-findings)
- [What worked well](#what-worked-well)
- [The code](#the-code)
- [Reproducing](#reproducing)

## Summary of findings

Ranked by how much they cost in this workload. Every finding has a repro in this repo and was checked against the compiler, the emitted LLVM IR or the docs. "Main" means a local build of a work-in-progress branch based on main at `cb22bf404` (2026-11-15), the newest build at hand.

| # | Finding | Severity | Measured impact | Status on main | Suggested fix |
|---|---|---|---|---|---|
| 1 | An async closure passed to a nested (local) function deep-copied its read-only captures on every call | High | f32 search 92 → 4.35 ms per query once avoided (21×); +42% instructions and +53 MB peak memory in a minimal repro | **Fixed** by #2734; verified: overhead +0.3%, no transfer in the IR | — |
| 2 | `bf16x8.dotAdd` quiets NaN payloads lane by lane before every FMA | High for bf16 users | 150 ns vs 38 ns (C f16) and 41 ns (C `bfdot`) per 384-dim dot | Still present; [#3129](https://github.com/velt-lang/velt/issues/3129) | Plain shift inside `dotAdd`; pick `bfdot` at run time |
| 3 | Bounds checks in SIMD loops only go away with `T.fits()`; the array data pointer is reloaded every iteration | Medium | f32 61 vs 46 ns (+34%) without `fits()`; int8 15.7 vs 11.0 ns in C (1.4×) | Still present (same int8 loop) | Loop versioning; hoist the buffer pointer; offset/length forms of std `dotI8`/`dotF16` |
| 4 | `Math` has no `exp`, `log`, `sin`, `cos`, `tanh`, `atan2`, `cbrt` or `log2`; the error says "not a static method" | Medium | Blocks softmax, GELU, most ML and geometry code | Still present; [#3130](https://github.com/velt-lang/velt/issues/3130) | Add them; until then a "not supported yet" error |
| 5 | Smaller: `f32` has no `toFixed`; `i64` needs `as number` for `Math.ceil`; no relaxed-FP reduction; 128-bit vectors only; `shared(x)` needs a copy to keep using `x` | Low each | Relaxed FP is 5.7× in C; `freeze()` copies 147 MiB | `toFixed` still missing; others planned | See [Smaller findings](#smaller-findings) |

## Existing issues and PRs

Searched in [velt-lang/velt](https://github.com/velt-lang/velt) on 2026-11-17, for each finding.

| Finding | Related | What it covers |
|---|---|---|
| 1. Capture copies | [#2726](https://github.com/velt-lang/velt/issues/2726) (open, design) | Exactly this: "async closures copy their captures where they cross threads, not on every call". It says the cost "hasn't been measured yet"; the numbers here are a measurement |
| | [#2734](https://github.com/velt-lang/velt/pull/2734) (merged 2026-11-11) | Step 1 of #2726: "a call through a function value is no thread boundary". Fixes this finding; not in 0.6.0 (released 2026-11-07) |
| 2. bf16 `dotAdd` | [#3129](https://github.com/velt-lang/velt/issues/3129) (filed from this report) | This finding, with a self-contained repro |
| | [#2387](https://github.com/velt-lang/velt/pull/2387) (merged) | Added `dotAdd`/`dotAddFast` on `f16x8` and `bf16x8` |
| | [#2049](https://github.com/velt-lang/velt/issues/2049) (open) | The f16/bf16 design, including the NaN-quieting contract |
| 3. Bounds checks | [#2485](https://github.com/velt-lang/velt/issues/2485) (open) | `fits_pair` follow-ups. Mentions the loop invariant not hoisted **on x86**; the arm64 pointer reload and the hand-written `i + 16 <= dim` case are not covered |
| | [#2335](https://github.com/velt-lang/velt/pull/2335) (merged) | `fits(a, j) && fits(b, j)` merged into one test |
| | [#1718](https://github.com/velt-lang/velt/issues/1718) (open) | Borrowed byte views checked per range, the kind of sub-range view the std dot products would need |
| 4. `Math` functions | [#3130](https://github.com/velt-lang/velt/issues/3130) (filed from this report) | Nothing existed before for `exp`/`log`/`sin`. [#1676](https://github.com/velt-lang/velt/issues/1676) and PR [#3086](https://github.com/velt-lang/velt/pull/3086) are about `Math.pow` |
| Relaxed FP, wider vectors | [#1866](https://github.com/velt-lang/velt/issues/1866) areas 8 and 10, [#1981](https://github.com/velt-lang/velt/issues/1981) rows 8 and 10 | Scoped `@fastmath` lowering to `reassoc contract`; `f32x8`/`f32x16` |
| Runtime CPU dispatch | [#1981](https://github.com/velt-lang/velt/issues/1981) step 5 | CPU features as a table, constant checks under `--cpu` |
| `f32.toFixed` | [#3039](https://github.com/velt-lang/velt/pull/3039) (open) | Number formatting leftovers; doesn't mention `f32` |

## Setup

All numbers are best-of-N on a heavily loaded machine, so treat absolute times as upper bounds and compare ratios.

| Item | Value |
|---|---|
| Machine | Apple M4 (4 performance + 6 efficiency cores), 32 GB, Darwin 27.0.0 |
| Toolchain | velt 0.6.0 (3282544, aarch64-apple-darwin), `--release`; Apple clang 21 for the C reference (`-O3 -mcpu=apple-m4`) |
| Data | 100,000 vectors × 384 dims in 1,000 clusters (uniform centroids plus uniform noise of amplitude 0.75); 20 queries drawn from the same clusters, not stored |
| Search | Brute force over unit vectors, top-10 by a bounded min-heap, one thread or split across `workerCount()` = 10 tasks |
| Ground truth | Exact top-10 from the scalar f32 search |
| Timing | Best of 3–7 rounds per path; kernel comparisons run Velt and C alternately, three times, keeping the best of each |

The load average was 48–93 during every measurement, because parallel Velt compiler builds were running on the same machine. Single-thread end-to-end times varied up to 4× between identical runs; recall and the Velt-vs-C ratios were stable. Instructions retired was tried as a load-proof measure, but it also counts the runtime's worker threads, so it is used only for finding 1's minimal repro, where both sides have the same threads.

## Results by storage format

int8 is the clear winner: a quarter of the memory, 3.7× faster than f32 on one core, and recall@10 of 0.975–0.99. f16 halves the memory with recall 0.99–1.000.

| Format | Bytes per vector | Index size (100k) | Best ms/query, 1 core | Best ms/query, 10 cores | recall@10 | top-1 |
|---|---|---|---|---|---|---|
| f32 scalar (reference) | 1536 | 147 MiB | 26.3 | — | 1.000 | 1.00 |
| f32 `f32x4` | 1536 | 147 MiB | 4.35 | 2.68 | 1.000 | 1.00 |
| f16 `f16x8.dotAdd` | 768 | 73 MiB | 2.69 | 2.04 | 0.99–1.000 | 1.00 |
| bf16 `bf16x8.dotAdd` | 768 | 73 MiB | 5.13 | 4.48 | 0.990–0.995 | 1.00 |
| bf16 `dotAddFast`, `--cpu native` | 768 | 73 MiB | 2.92 | 1.93 | 0.97–0.995 | 1.00 |
| int8 `i32x4.dotAdd` | 388 (384 + f32 scale) | 37 MiB | 1.17 | 0.95 | 0.975–0.99 | 0.90–1.00 |

The 10-core column understates the speedup. With load averages of 48–93 there were no idle cores, and an earlier run at a load of about 30 had f32 at 3.3 ms against 30 ms on one core (about 9×). On an idle machine the parallel scans should be limited by memory bandwidth instead.

`--cpu native` changed nothing for f32, f16 or int8 on the M4, because the default arm64 build already emits `fmlal` and `sdot`. Only bf16 `dotAddFast` gains (it becomes `bfdot`).

## Kernels against C NEON

The f32 kernel matches C. f16 and int8 are 1.2–1.4× behind because of loop overhead (finding 3), and bf16 `dotAdd` is 3.7× behind (finding 2).

Each line is one query dotted against 5,000 rows (7.7 MB, in cache), with no top-k, in ns per 384-dim vector. Each number is the best of 45 runs spread over three interleaved passes. The 100k-row runs gave the same ratios.

| Kernel | C, hand-written NEON | Velt | Velt ÷ C |
|---|---|---|---|
| f32, 4 accumulators, FMA | 43.2 (`vfmaq_f32`) | 45.6 default, 43.8 native (`f32x4.mulAddFast`, loop on `fits()`) | 1.01–1.06 |
| f32, same kernel, loop on `i + 16 <= dim` | — | 61.0 | 1.41 |
| f32, plain scalar loop, strict FP | 285.6 | 247.5 | 0.87 |
| f32, plain loop, reassociation allowed | 43.6 (auto-vectorized) | no way to ask for it | — |
| f16 × f16 → f32 | 38.4 (`fmlal`/`fmlal2`) | 45.5 (`f16x8.dotAdd`) | 1.18 |
| bf16 × bf16 → f32, exact | — | 150.4 (`bf16x8.dotAdd`) | 3.7 vs `bfdot` |
| bf16, CPU dot product | 41.0 (`bfdot`) | 55.1 (`dotAddFast`, `--cpu native`); 150.4 without | 1.34 |
| int8 × int8 → i32 | 11.0 (`sdot`) | 15.7 (`i32x4.dotAdd`) | 1.43 |
| int8, std `dotI8` over one long array | — | 10.5–10.7 | 0.97 |
| f16, std `dotF16` over one long array | — | 32.2 | 0.84 |

The std functions reach C speed but only take whole arrays, so they can't score row *i* of a flat store without copying the row out first. The kernels in `src/quant.velt` had to be written by hand for that reason.

## Finding 1: closures copied their captures on every call (fixed on main)

In 0.6.0, an async closure that only reads a captured object, passed to a function declared inside another function, deep-copies that object every time it is called. Passed to a top-level function, the same closure does not. Nothing in the source, the docs or the diagnostics shows the copy.

**Status.** This is [#2726](https://github.com/velt-lang/velt/issues/2726), fixed by [#2734](https://github.com/velt-lang/velt/pull/2734) ("a call through a function value is no thread boundary"), merged 2026-11-11, four days after 0.6.0. Re-run on main (a local branch build based on main at `cb22bf404`, 2026-11-15): the nested call is within 0.3% of the direct call, peak RSS is unchanged, and `velt_rt_xfer_begin` no longer appears in the IR. The numbers below are 0.6.0's; #2726 notes the cost "hasn't been measured yet", so they may still be useful there.

**How it showed up.** The format benchmark first measured f32 search at 92.3 ms per query, against a 4.6 ms estimate from the raw kernel (46 ns × 100k rows). Its `measure` helper was an `async function` nested inside `main` that received `async (q) => index.search(q, k)`. Moving `measure` to the top level and passing its state as a parameter brought the same search to 4.35 ms.

**Minimal repro** (`bench/callshape.velt`): 200 searches over one 100k × 384 index, one calling style per process, measured with `/usr/bin/time -l`:

| Calling style | 0.6.0: instructions per search | 0.6.0: peak RSS | main: instructions per search* |
|---|---|---|---|
| `index.search(q, 10)` directly | 84.7 M | 427 MB | 84.5 M |
| closure passed to a top-level `async function` | 84.6 M | 427 MB | 84.4 M |
| same closure passed to an `async function` declared inside `main` | **120.2 M (+42%)** | **480 MB** | 84.8 M (+0.3%) |

\* Main was built with a debug runtime library, so its absolute counts aren't comparable with 0.6.0's; the comparison within each column is.

**Evidence in the 0.6.0 IR** (`velt build --release --emit llvm`). The closure handed to the nested function (`_V4mainN17_7bclosure_232_7d`) calls `velt_rt_xfer_begin`, then `_Gobjclone_…VectorIndex` on its captured index on every invocation, before boxing its future. The closure handed to the top-level function (`closure_230`) calls the clone only from its `_Genv_clone`/`_Genv_transfer` helpers. I did not confirm which fields the clone copies; the +53 MB peak suggests not the whole 147 MiB array.

**Semantics were fine.** `bench/capture_semantics.velt` mutates the captured object through both shapes, and both print what JavaScript semantics require (`1 2`, `3 4`) in debug and release. So it was a cost, not a wrong result.

## Finding 2: bf16x8.dotAdd is 3–4× slower than f16

Widening bf16 should be the cheapest conversion there is (a 16-bit shift). Yet `bf16x8.dotAdd` takes 150 ns per 384-dim dot, against 45.5 ns for `f16x8.dotAdd` and 41 ns for C `bfdot`. End to end, bf16 search is 1.9× slower than f16 on one core (5.13 vs 2.69 ms). Still present on main.

**Cause, from the IR.** `f32x4_dot_add_f16x8` is two intrinsics, `fmlal` and `fmlal2`. `f32x4_dot_add_bf16x8` widens each of its four half-vectors through `velt.bf16_to_f32.v4`, which quiets NaNs lane by lane:

```llvm
%x     = shl <4 x i32> %w, 16
%a     = and <4 x i32> %x, 0x7fffffff
%nan   = icmp ugt <4 x i32> %a, 0x7f800000
%mask  = sext <4 x i1> %nan to <4 x i32>
%quiet = and <4 x i32> %mask, 0x00400000
%r     = or  <4 x i32> %x, %quiet
```

That's about 5 extra vector operations per conversion and 4 conversions per call. On top of that, the second of its two FMAs adds into the first's result, so a 2-accumulator loop is bound by FMA latency.

`dotAddFast` falls back to this same path unless the program is built with `--cpu native` (or another `--cpu` with FEAT_BF16). So a portable binary on an M2 or later never uses `bfdot`, though SHA, AES and carry-less multiply already choose their instruction at run time.

**Suggested fixes.**

- Widen with a plain shift inside `dotAdd`. The FMA already turns a signalling NaN into a quiet one and, on arm64 with default-NaN off, keeps its payload, so the explicit quieting likely changes nothing in the result. This is worth checking against the documented NaN contract.
- Evaluate `bfmlalb`/`bfmlalt` (FEAT_BF16) for the exact path, if their rounding matches `dotAdd`'s documented `fma(fma(…))` order.
- Pick `bfdot` for `dotAddFast` at run time, as `hasAes()` and `hasShaArm()` do.

## Finding 3: bounds checks and pointer reloads in SIMD loops

A SIMD loop only loses its load checks when its condition is written as `T.fits(xs, i + lastOffset)`, as the docs advise. The obvious `i + 16 <= dim` keeps a check on every load, which makes the f32 kernel 34% slower (61.0 vs 45.6 ns). Even with `fits()`, each iteration carries work C doesn't. The int8 loop below is unchanged on main.

**The int8 inner loop on arm64** (from `otool -tv`, 2 `sdot` per iteration):

```asm
add   x0, x12, x17        ; aAt + i
cmp   x0, x8              ; fits(a, …)
ccmp  x17, x8, #0x2, ls   ; fits(b, …)
b.hi  …
sub   x0, x0, #0x10
cmp   x0, x8
b.hi  …                   ; a second range check per iteration (my reading)
ldr   x0, [x15]           ; array data pointer, reloaded every iteration
add   x0, x0, x17
add   x1, x0, x12
ldp   q2, q3, [x1, #-0x10]
ldp   q4, q5, [x0, #-0x10]
sdot.4s v1, v2, v4
sdot.4s v0, v3, v5
add / add / mov / cmp / b.ls
```

That's about 13 instructions of bookkeeping for 2 `sdot`, which explains int8 at 15.7 ns against 11.0 in C. The arrays aren't written in the loop, so the data-pointer load could be hoisted. [#2485](https://github.com/velt-lang/velt/issues/2485) item 2 reports the x86 side of this (`len >= 16` not hoisted).

**Suggested fixes.**

- Loop versioning: when the loop's bound and offsets are linear in `i`, check `aAt + dim <= a.length` (and the same for `b`) once before the loop and run a check-free body. Hand-written `i + 16 <= dim` conditions would then be as fast as `fits()`.
- Hoist the array buffer pointer out of loops that don't resize the array (mark the load invariant, or attach the `noalias`/`invariant.load` metadata LLVM needs).
- Give std's `dotI8`, `dotF16` and `dotBf16` an offset/length form (`dotI8(a, aAt, b, bAt, len)`), or add a zero-copy sub-range view of a `T[]` (as [#1718](https://github.com/velt-lang/velt/issues/1718) proposes for bytes). The std versions already reach C speed (10.6 ns) but only take whole arrays.

## Finding 4: Math has no exp, log or trig functions

The prelude's `Math` has `sqrt`, `floor`, `ceil`, `round`, `trunc`, `abs`, `sign`, `pow`, `max`, `min`, `hypot`, `imul`, `clz32`, `umulh` and `random`, and nothing else. `velt check` rejects all of these, in 0.6.0 and on main:

```ts
console.log(Math.exp(x), Math.log(x), Math.sin(x), Math.tanh(x),
            Math.atan2(x, x), Math.cbrt(x), Math.log2(x));
```

```text
error: `Math.exp` is not a static method
error: `Math.log` is not a static method
… (one per function)
```

I hit this writing the first tests, whose data used `Math.sin` and `Math.cos`. For a language with `bf16`, int8 dot products and an ML focus ([#1866](https://github.com/velt-lang/velt/issues/1866)), `exp` and `log` (softmax, log-likelihoods), `tanh` (GELU) and trig (positional encodings, geometry) are table stakes, and every TypeScript user will expect them.

**Suggested fix.** Add the remaining ES `Math` functions, through the math-operation registry [#1981](https://github.com/velt-lang/velt/issues/1981) built (LLVM's `llvm.exp`/`llvm.log`/`llvm.sin`, or libm). Until then, make the error say what's true, for example "`Math.exp` is not supported in Velt yet", instead of "not a static method", which reads as a typo on the user's side.

## Smaller findings

Each is small. The first two caused most of the compile errors I hit while writing the project.

| Area | What happened | Suggestion | Tracked |
|---|---|---|---|
| `f32` formatting | `score.toFixed(3)` on an `f32`: "no method named `toFixed` found for type `f32`". `toFixed`, `toExponential` and `toPrecision` exist only on `f64`. Still on main | Give `f32` the `f64` number methods (the conversion is exact) | — |
| `i64` in float math | `Math.ceil(n / parts)` with `i64` operands: "expected f64, found i64"; needed `Math.ceil((n as number) / (parts as number)) as i64`. The same for `Math.floor(random() * n)`. The message is clear; the boilerplate is heavy | Integer overloads of `Math.ceil`/`floor`/`round`, or `ceilDiv` in `velt/math` | — |
| Relaxed FP | A plain `for` dot-product loop stays scalar (247 ns) because the sum can't be reordered. C with `#pragma clang fp reassociate(on)` auto-vectorizes the same loop to 43.6 ns (5.7× faster than Velt's) | A scoped `@fastmath` or a reassociating reduction in std | #1866 area 10, #1981 row 10 |
| Vector width | Only 128-bit types. Native on Apple; on x86 it leaves AVX2 and AVX-512 unused (not measured here) | Width-generic vectors, as Zig's `@Vector(N, T)` or Rust's `Simd<T, N>` | #1866 area 8, #1981 row 8 |
| Sharing without a copy | `shared(x)` takes `x` itself, so an index that keeps its rows must hand over `shared(this.data.clone())`: 147 MiB copied in 30–98 ms | Turn an owned array into a `shared` read-only one and keep reading it through the shared handle | — |
| Runtime CPU dispatch | `--cpu native` on the M4 changed nothing but bf16 `dotAddFast`. On x86 the docs say FMA and F16C need `--cpu` too | Choose these at run time, as SHA and AES are | #1981 step 5 |

## What worked well

- **The f32 SIMD kernel runs at C speed.** `f32x4` with `mulAddFast` and a `fits()` loop came within 1–6% of hand-written `vfmaq_f32`, and the default arm64 build already emits `fmlal` and `sdot`.
- **Parallelism was about 20 lines.** `shared(...)` rows plus one `spawn` per worker, with races checked at compile time. Results were identical across scalar, SIMD and parallel paths for 20 of 20 queries.
- **Quantized formats were easy to add.** `velt/half` (`narrowF16`, `narrowBf16`) and the `dotAdd` family made f16, bf16 and int8 storage a 255-line addition, with recall at 0.975–1.000.
- **The compiler messages tell you the fix.** For example, "`i64` converts to `number` only with `as number` (not every value is exact)".
- **The docs were enough.** The project was written from `llms.txt` and the `.md` docs pages alone. They're precise down to which instruction each method becomes, which made the IR and disassembly checks quick.
- **Tooling:** `velt new`, `velt test` (11 tests in about 2 s), `velt fmt`, release builds of about 1.6 s with a warm cache, a 1.4 MB binary, and `--emit llvm` for checking codegen.

## The code

```sh
velt run --release                                   # text demo + f32 benchmark
velt run --release -- --query "how do threads share memory"
velt test
```

- Vectors are normalized on insert, so cosine similarity is a plain dot product.
- f32 uses `f32x4`: 16 floats per step across 4 accumulators with fused multiply-add.
- f16 and bf16 use `f16x8`/`bf16x8.dotAdd` (or `dotAddFast`).
- int8 uses symmetric per-vector quantization with `i32x4.dotAdd`.
- A bounded min-heap keeps the top-k results.
- The multi-core search puts the rows in `shared(...)` memory and `spawn`s one scan per worker.
- The text demo embeds text with the hashing trick (FNV-1a over words and character trigrams), a stand-in for real model embeddings.

| File | What |
|---|---|
| `src/vec.velt` | f32 SIMD dot product, scalar reference, norm, normalize, cosine |
| `src/quant.velt` | f16 / bf16 / int8 dot kernels, int8 quantization |
| `src/topk.velt` | bounded min-heap of the best hits |
| `src/index.velt` | `VectorIndex` (f32, one thread) and `FrozenIndex` (f32, all cores) |
| `src/compressed.velt` | `CompressedIndex`: f16 / bf16 / int8 copies, one thread or all cores |
| `src/embed.velt` | hashing-trick text embeddings |
| `src/main.velt` | text demo and f32 benchmark |
| `tests/*.test.velt` | 11 tests: SIMD = scalar for every tail length, quantized kernels exact or within rounding, top-k, parallel = single thread |

## Reproducing

| File | Reproduces |
|---|---|
| `bench/quant.velt` | Results by storage format (speed, recall@10, top-1); `--only <path>` runs one path |
| `bench/kernels.velt`, `bench/kernels.c`, `bench/run_kernels.sh`, `bench/best.py` | Kernels against C NEON, interleaved, best per kernel |
| `bench/callshape.velt` | Finding 1: instructions and peak RSS per calling style |
| `bench/overhead.velt` | Finding 1: wall-clock time of the same calling styles |
| `bench/capture_semantics.velt` | Finding 1: mutation through both closure shapes stays visible |

```sh
velt test
velt run --release bench/quant.velt -- --n 100000 --dim 384 --queries 20 --rounds 7
velt run --release --cpu native bench/quant.velt

mkdir -p target
clang -O3 -mcpu=apple-m4 bench/kernels.c -o target/kernels_c
velt build --release bench/kernels.velt -o target/kernels_velt
velt build --release --cpu native bench/kernels.velt -o target/kernels_velt_native
sh bench/run_kernels.sh 3 > target/k.txt && python3 bench/best.py target/k.txt

velt build --release bench/callshape.velt -o target/callshape
for m in none direct param nested; do /usr/bin/time -l target/callshape $m; done

velt build --release --emit llvm bench/kernels.velt > target/kernels.ll   # finding 2: velt.bf16_to_f32.v4
otool -tv target/kernels_velt | grep -n -B14 sdot                         # finding 3
```

The C reference needs an arm64 CPU with FEAT_FHM, FEAT_BF16 and FEAT_DotProd (Apple M2 or later). Run on an idle machine for absolute numbers; the ones here were taken at load averages of 48–93.
