# Six attempts on the codebook matvec, one of which worked, 2026-09-20

Decode on the two models built from the IQ types was the largest end-to-end gap left: the 27B at 0.84 of
Vulkan and the MoE at 0.82, where every other model sits between 0.93 and 0.99
([`logs/fedora44-campaign-final-defaults-2026-09-20/`](../fedora44-campaign-final-defaults-2026-09-20/)).
This is what was tried, in the order it was tried, with the four failures kept because each one says
something about what the kernel is actually limited by. The fifth and sixth were made a day later, and
the sixth overturned the reading the first five had settled on; they are at the end.

## Where the models sit

Weight bytes read per second during decode, from the campaign medians and the model file sizes, against
the 432 GB/s streaming-read peak the board reaches
([`logs/membw-2026-08-19/`](../membw-2026-08-19/)); and the same rate expressed as weight values decoded
per second, which is what the kernel's instruction count is spent on. The MoE is left out because only
part of its weights are read per token.

| model | GiB | t/s | GB/s | of peak | Gvalue/s | Vulkan t/s | Vulkan Gvalue/s |
|---|---|---|---|---|---|---|---|
| qwen2.5-1.5B q4_K | 1.04 | 198.1 | 221 | 51 % | 393 | 212.4 | 422 |
| qwen3-8B q8_0 | 8.24 | 38.5 | 341 | 79 % | 341 | 39.0 | 345 |
| deepseek-14B q4_K | 8.37 | 32.3 | 290 | 67 % | 516 | 34.9 | 558 |
| qwen3-14B q4_K | 8.63 | 32.3 | 299 | 69 % | 532 | 34.6 | 570 |
| qwen3.8-27B IQ3 | 11.09 | 14.8 | 176 | 41 % | 403 | 17.6 | 479 |

The q8_0 model reads at 79 percent of the board's peak and comes within 1.3 percent of Vulkan, so the
matvec can be memory-bound when the weights are cheap to decode. The 27B reads at 41 percent of peak
against the 14B models' 67 to 69, which looks like a great deal of headroom, but per value decoded it is
only 23 percent behind them. Its problem is that 3.5 bits a value means more values to decode per byte
of model, and Vulkan decodes them 19 percent faster than we do.

## The instruction mix

`hipcc -S --cuda-device-only` on the shipped kernel, bucketed by opcode class
([`isa-mix-shipped.log`](isa-mix-shipped.log), from [`scripts/isa_mix.py`](../../scripts/isa_mix.py)).
Vector ALU instructions in the whole kernel body divided by its fused multiply-adds, for the variant
with bias/gate fusion. The absolute figure is inflated by prologue, epilogue and the 64-bit address
arithmetic every variant shares, so it is the comparison between types that carries the meaning:

| type | VALU per FMA | of which conversion and sign |
|---|---|---|
| q4_K | 6.3 | 2.2 |
| q8_0 | 6.4 | 1.6 |
| iq2_xxs | 8.2 | 2.3 |
| iq3_xxs | 8.3 | 2.3 |
| iq3_s | 10.4 | 2.7 |

A codebook entry costs about nineteen instructions for four values: two to build the index, one load,
four `v_cvt_f32_ubyte` and eight to apply the sign bits, for four `v_fmac_f32`. iq3_s also drains the
memory counter seven times against q8_0's two, because the codebook lookup is a second dependent load
whose address comes from bytes that were themselves just loaded.

## What was tried

All four were run as run-time switches inside one build, so each arm uses the same binary and the same
`libggml-hip.so`; arms alternate within a round and the figures are medians of three rounds of three
samples. Spreads are given with each table. The switches were a scaffold and are not in the shipped
kernel: three of the four arms lost, and the one that won is now the only behaviour, so
[`patches/llamacpp/0009-rdna1-iq-matvec-shmem-codebook.patch`](../../patches/llamacpp/0009-rdna1-iq-matvec-shmem-codebook.patch)
carries no environment variables at all.

### 1. The codebook in packed fp16: 15 percent slower

The board's `v_pk_fma_f16` rate is three times its `v_fma_f32` rate, so the codebooks were converted to
half2 pairs by a small init kernel, the four sign bits of a group made to index a table of xor masks,
and the sixteen products accumulated in one half2 promoted to f32 once per superblock. Grid bytes reach
62, so entries were stored divided by 64, exactly, and the scale folded back into the block scale. All
1186 `test-backend-ops` MUL_MAT cases still passed.

| model | float | packed fp16 | ratio |
|---|---|---|---|
| qwen3.8-27B tg64 | 14.80 | 12.59 | 0.850 |
| qwen3.6-35B MoE tg64 | 70.08 | 67.31 | 0.960 |
| qwen3.8-27B pp512 | 81.20 | 81.20 | 1.000 |
| qwen3.6-35B MoE pp512 | 340.65 | 340.84 | 1.001 |

Spread 0.9 to 1.1 percent on decode; prefill is the control, and it does not move, which confirms the
switch reaches only the matvec. The kernel had 8 percent fewer instructions and ran 15 percent slower.
It also needed 8 to 22 more registers and one extra dependent load for the sign mask
([`isa-registers.log`](isa-registers.log)), and those cost more than the arithmetic saved.

### 2. Forcing occupancy: no better at best, 64 percent worse at worst

The IQ kernels use 70 to 104 registers against q8_0's 26, which caps them at nine to fourteen resident
waves where q8_0 gets twenty. `__launch_bounds__(32, n)` was templated so the compiler could be told to
leave room for 8, 12 or 16.

| model | default | min 8 | min 12 | min 16 |
|---|---|---|---|---|
| qwen3.8-27B tg64 | 14.78 | 14.74 | 13.34 | 5.33 |
| qwen3.6-35B MoE tg64 | 69.90 | 69.94 | 69.94 | 61.35 |

Spread 0.1 to 2.1 percent. A floor of 8 changes no register count and so changes nothing. Above that the
compiler buys waves with scratch: 104 bytes a thread at 12 and 176 at 16 for iq3_s with fusion. Waves
bought that way are worth less than the spills cost, by a wide margin.

### 3. Rows per block: two is already the best of one, two and four

Each 32-lane block covers two rows, so the sixteen activations a thread loads are shared by two rows.

| model | one row | two rows | four rows |
|---|---|---|---|
| qwen3.8-27B tg64 | 10.14 | 14.86 | 14.43 |
| qwen3.6-35B MoE tg64 | 69.93 | 70.05 | 68.10 |

Spread 0.1 to 0.7 percent. Halving the sharing costs 32 percent on the 27B, which says activation
traffic matters a great deal; doubling it costs 3 percent, because the extra live state costs more
waves than the saved traffic is worth. The shipped value sits on top of the curve.

### 4. The codebook in shared memory: 1.8 percent, and kept

ggml-vulkan's IQ matvec shaders call `init_iq_shmem` before their loop, which copies the codebook into
shared memory so that every lookup is an LDS read, not a global gather
(`ggml/src/ggml-vulkan/vulkan-shaders/types.glsl`). Ours gathered from global memory. A 32-lane block
copies 512 dwords, 256 for iq3_xxs, in sixteen loads per thread, and then performs about eighty lookups
per thread on a 5120-column matrix, so the copy pays for itself many times over. Alongside it, the four
codebook indices a thread needs was consecutive bytes at an even offset, as are iq3_s's two
sign bytes, so three 16-bit loads replace six 8-bit ones.

| model | shipped | 16-bit loads | shared memory | both |
|---|---|---|---|---|
| qwen3.8-27B tg64 | 14.76 | 14.72 | 14.59 | 15.01 |
| qwen3.6-35B MoE tg64 | 69.25 | 69.81 | 66.78 | 66.22 |

Spread 0.2 to 0.8 percent. Neither half helps alone and the pair helps only the dense model: staging
costs one copy per block, and the expert-id path launches a block per used expert, which covers too few
superblocks to earn it back. Confining the staging to ordinary matrix-vector products fixes that:

| model | shipped | both, expert path excluded |
|---|---|---|
| qwen3.8-27B tg64 | 14.78 | 15.04 |
| qwen3.6-35B MoE tg64 | 69.94 | 70.09 |

Spread 0.1 to 0.3 percent. That is what patch 9 does.

## Rows per block, measured again after the staging

Patch 9 changed the register count of the IQ kernels, iq3_s from 70 to 64 without fusion and 104 to 89
with it, and the rows sweep above was taken before that. Three exclusions decided under an old parameter
have since reversed when the parameter changed, so this one was measured again on the shipped build
(`rows-retest/`, medians of three rounds of three samples, arms interleaved):

| model | one row | two rows | four rows |
|---|---|---|---|
| qwen2.5-1.5B | 198.07 | 193.89 | 187.04 |
| qwen3.6-35B MoE | 71.35 | 69.76 | 66.16 |
| qwen3.8-27B | **12.81** | 15.04 | 14.18 |

Spread at most 1.2 percent. The staging did move it: one row per block was 32 percent behind on the 27B
before and is 15 percent behind now, and on the MoE it went from level to 2.3 percent ahead. But no
single value wins everywhere, so two rows stays. One row is worth 2.2 percent on the 1.5B and 2.3 on the
MoE and costs the 27B 15, which is what halving the sharing of the activations should do on the model
with the longest rows, K of 5120 against 1536 and 2048. A rule on K would capture that, and it is not
taken here: it would be fitted to three models.

## 5. The codebook in packed fp16 again, this time in shared memory: 19 percent slower

Attempt 1 put the half2 codebook and the sign masks in global memory and lost 15 percent, and the
reading at the time was that it paid for a second dependent global load per entry and up to twenty-two
registers. Patch 9 then gave the block a shared-memory codebook, so the obvious repair was to stage the
half2 form there instead: no global tables, no init kernel, no second global load, the conversion done
once while staging. It is worse, not better (`h2-in-lds/`):

| model | codebook as bytes | as half2 in shared memory | ratio |
|---|---|---|---|
| qwen3.8-27B tg64 | 15.10 | 12.29 | 0.814 |
| qwen3.6-35B MoE tg64 | 69.89 | 70.14 | 1.004 |

Spread 0.3 to 1.3 percent. The MoE does not move because its codebook work all arrives through the
expert path, which does not stage, so the switch cannot reach it; that is a useful control on the
switch. The 27B loses 18.6 percent where the global version lost 15.

The registers say why, and they say the first reading was wrong
([`h2-in-lds/isa-registers.log`](h2-in-lds/isa-registers.log)):

| kernel | bytes | half2 | waves per SIMD |
|---|---|---|---|
| iq3_s | 64 | 72 | 16 to 14 |
| iq3_s, fused | 89 | 108 | 11 to 9 |
| iq2_xxs, fused | 73 | 92 | 14 to 11 |
| iq3_xxs, fused | 73 | 84 | 14 to 12 |

The packed form costs 8 to 19 registers with the tables in shared memory, against 8 to 22 with them in
global. **The registers were never the tables.** They are the live state the packed form needs, the
eight half2 activations and the accumulator carried across the row loop, and moving a table does not
touch that. So this closes the direction instead of one implementation of it: the arithmetic saving is
real, the kernel is register-limited, and on this kernel the second always beats the first.

## 6. Eight live activations instead of sixteen: 16 percent slower, and it overturns the reading

Attempts 1 and 5 both cost registers, so the reading after five was that the kernel is register-limited.
ggml-vulkan's shader holds eight activations at a time where ours holds sixteen, which is the one
structural difference left, so this splits the superblock into halves: each half reads only the two
codebook entries and the one sign byte it needs, and the two float4 of activations it multiplies.

The registers fall a long way, further than expected ([`halves/isa-registers.log`](halves/isa-registers.log)):

| kernel | sixteen | eight | waves per SIMD |
|---|---|---|---|
| iq3_s, staged, fused | 89 | 77 | 11 to 13 |
| iq3_s, staged | 64 | 58 | 16 to 17 |
| iq2_xxs, staged, fused | 73 | 58 | 14 to 17 |
| iq3_xxs, staged, fused | 73 | 52 | 14 to 19 |
| iq3_xxs, staged | 61 | 43 | 16 to 20 |

And it is slower (`halves/`):

| model | sixteen | eight | ratio |
|---|---|---|---|
| qwen3.8-27B tg64 | 15.08 | 12.63 | 0.837 |
| qwen3.6-35B MoE tg64 | 69.01 | 68.29 | 0.990 |

Spread 0.2 to 1.2 percent. Two more waves per SIMD on iq3_s and five on iq3_xxs, and sixteen percent
worse. **So the kernel is not register-limited either**, and the reading after five attempts was wrong.
What halving the activations also did was double the block header reads: `d`, the scale byte, `qh` and
the sign bytes are now fetched once per eight values instead of once per sixteen.

## What the six together say

One sign runs through all six, and it is neither instructions nor registers. **Every change that added
memory operations lost, and the only change that removed one won.** Packed fp16 in global memory added a
dependent load per entry and lost 15 percent; the same in shared memory lost 19; forcing occupancy
bought waves with spills and lost up to 64; one row per block doubled the activation traffic and lost
15; eight live activations doubled the header reads and lost 16, while handing back up to five waves per
SIMD. The one that shipped removed a dependent global load per group and gained 1.8 percent. Instruction
count moved the wrong way twice and register count once.

The count predicts the sign but not the size, and the two points that bracket it are worth writing down
before anyone spends another day here. Counting memory operations per row per sixteen values, the
shipped kernel issues twelve: six weight loads, four codebook reads from shared memory, and two of the
four activation vectors, which two rows share. Patch 9 took that from fifteen to twelve, and the global
loads among them from thirteen to six, for 1.8 percent. Attempt 6 took it from twelve to eighteen and
cost 16. So the response is real, strongly asymmetric, and nowhere near linear: more than halving the
global loads bought under two percent.

That prices the one direction left. ggml-vulkan reads its block header once per thirty-two values where
this reads it once per sixteen, and restructuring to match would take the count from twelve to ten and a
half, a twelve percent cut. Against patch 9's twenty percent cut for 1.8 percent, that is worth about
one percent, for a rewrite of all three type branches and the loop around them. It is not taken, and
this arithmetic is the reason, not fatigue after six attempts.

The remaining distance to Vulkan on the 27B, about 17 percent, is in the matvec kernel itself and not
between dispatches; see the correction at the top of
[`logs/kerntrace-2026-09-19/`](../kerntrace-2026-09-19/), where the apparent 18 percent launch gap turned
out to be the tracer's own overhead.

## Correctness

The q4_K, q5_K, q6_K and q8_0 kernels come out of the compiler instruction for instruction identical to
the shipped build, so the four models that use no codebook cannot be affected; only the three IQ kernels
differ. `test-backend-ops` passes 1186/1186 MUL_MAT and 865/865 MUL_MAT_ID cases with the staged path. The
transformation reads the same codebook words in the same order and leaves the arithmetic untouched, and
greedy generation on both models is token-identical to the output recorded for the earlier build in
[`logs/rdna1-mmvq-2026-09-18/f32iq/`](../rdna1-mmvq-2026-09-18/f32iq/); only the throughput line differs
([`text-27b-staged.txt`](text-27b-staged.txt), [`text-moe-staged.txt`](text-moe-staged.txt)).

## Files

`isa-mix-shipped.log` is the opcode histogram of the shipped kernel and `isa-registers.log` the register,
scratch and wave counts of every variant compiled during the investigation. The `llama-bench` JSONL is
named for its experiment: `h2-` the packed-fp16 arm, `occ-` the occupancy sweep, `rows-` the row count,
`lds-` the shared-memory and 16-bit-load combinations, `gate-` the same with the expert path excluded,
`rows-retest/` the row count measured again on the shipped build, `h2-in-lds/` the packed-fp16 codebook
staged in shared memory, and `halves/` the eight-activation split.
