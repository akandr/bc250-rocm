# Where the MoE's token goes, decode and prefill, 2026-09-20

With the codebook matvec settled ([`logs/rdna1-iq-matvec-2026-09-20/`](../rdna1-iq-matvec-2026-09-20/))
the qwen3.6-35B-A3B MoE is the weakest of the six models, decoding at 0.81 of Vulkan where the next
worst is 0.86, and prefilling at 0.74. This is where both of its tokens go.

## The kernels

[`scripts/kerntrace.cpp`](../../scripts/kerntrace.cpp) on `llama-bench -p 0 -n 64 -r 1`, nine-patch
build. The tool now takes `KERNTRACE_FULL=1`, which keeps the template arguments instead of stripping
them, because six instantiations of the same matvec are otherwise indistinguishable in its table.
83060 dispatches, 776 ms of kernel time, 1140 ms from first dispatch to last completion
([`trace-moe-tg-graphs-on.txt`](trace-moe-tg-graphs-on.txt)):

| dispatches | per token | ms | share | avg us | kernel |
|---|---|---|---|---|---|
| 5670 | 88.6 | 158.9 | 20.5 % | 28.0 | float matvec q5_K |
| 2520 | 39.4 | 104.4 | 13.5 % | 41.4 | float matvec iq2_xxs, fused |
| 2331 | 36.4 | 73.8 | 9.5 % | 31.7 | float matvec iq3_xxs |
| 4410 | 68.9 | 60.3 | 7.8 % | 13.7 | float matvec q6_K |
| 63 | 1.0 | 57.7 | 7.4 % | 915.5 | float matvec q4_K, the output head |
| 8820 | 137.8 | 46.3 | 6.0 % | 5.3 | float weights matvec (`mul_mat_vec_f`) |
| 3150 | 49.2 | 44.0 | 5.7 % | 14.0 | float matvec q5_K, fused |
| 2520 | 39.4 | 25.5 | 3.3 % | 10.1 | `topk_moe_cuda` |
| 1890 | 29.5 | 21.1 | 2.7 % | 11.2 | `k_get_rows_float_vec` |
| 1890 | 29.5 | 17.8 | 2.3 % | 9.4 | `gated_delta_net_cuda` |
| 5103 | 79.7 | 17.5 | 2.3 % | 3.4 | `rms_norm_f32` |

Matrix-vector work is 70 percent of it, as on the other models, but the split inside that is not what the
model's name suggests: the q5_K attention projections are 26 percent and the IQ2/IQ3 experts 23. The
output head is one 915 us kernel a token, 7.4 percent on its own. Every IQ instantiation here is
unstaged, which is right: this model's codebook work all arrives through MUL_MAT_ID, the one path patch 9
deliberately leaves alone.

## Per op, ROCm is not behind

A one-token graph exported with `test-export-graph-ops -b 1 -ub 1` and replayed on both backends
([`ops-moe-n1.txt`](ops-moe-n1.txt), [`opcmp-moe-n1.txt`](opcmp-moe-n1.txt) from
[`scripts/opcmp.py`](../../scripts/opcmp.py)). Of 63 shared ops, summed: 1454 us on ROCm against 1542 on
Vulkan. The largest, the output head, is 955 us against 921, four percent behind. Nothing is far behind:

| ROCm us | Vulkan us | ratio | op |
|---|---|---|---|
| 955.1 | 920.5 | 1.04 | MUL_MAT result_output |
| 16.5 | 8.1 | 2.03 | ARGSORT ffn_moe_argsort |
| 34.1 | 25.9 | 1.31 | MUL_MAT_ID ffn_moe_down |
| 24.7 | 20.9 | 1.18 | MUL_MAT z |
| 24.9 | 21.2 | 1.17 | MUL_MAT attn_output |
| 41.0 | 37.7 | 1.09 | MUL_MAT node_13 |

The sum is not a token and must not be read as one: each replayed op carries its own submission
overhead, and the export keeps one instance of each unique op, so a shape that occurs in forty layers
counts once. The ARGSORT row is a decoy for a second reason: the export lists graph nodes, and in the
real graph llama.cpp fuses the expert selection into `topk_moe_cuda`, so no argsort kernel
appears in the trace at all.

> **Corrected 2026-09-23.** This page went on to read that sum as "ROCm is at or ahead of Vulkan on
> every shape it contains, so no kernel is slow". That does not hold, for two reasons, and
> [`logs/moe-kernels-reweighted-2026-09-23/`](../moe-kernels-reweighted-2026-09-23/) works both
> through. **ROCm is faster on 30 of the 63 shapes and slower on 33**, median ratio 1.024, so "every
> shape" was never true in either direction. And the 88-microsecond margin in the sum above is
> produced by **one artifact row**: Vulkan's `MUL_MAT shared_expert_gate-0`, a 1x1 output, replays at
> 131.71 us, where the real graph dispatches that shape as `MUL_MAT_VEC f32 m=1 n=1 k=2048` at 4.26 us,
> forty times a token. Correcting that row alone turns the sum from 0.943 to 1.028 against ROCm.
> The sign is not scattered either: splitting the replayed matmuls by what the weights are, **ROCm is
> slower on 9 of the 10 quantised ones**, median ratio 1.078, and faster on all 3 f32 ones, median
> 0.820. So this is the same quantised-matvec gap as the 27B's, not anything new. How many
> microseconds of the deficit that is worth cannot be stated: ROCm and Vulkan do not issue the same
> matmul dispatches here, 437.7 matvec dispatches a token against 511, because ROCm fuses the expert
> gate/up pair that Vulkan runs separately. A kernel deficit does exist and this replay was hiding it;
> its size is open.

## Dispatch count is what separates the models

| model | dispatches per token | mean kernel | token | at the measured 1.78 us a dispatch |
|---|---|---|---|---|
| qwen2.5-1.5B | 375 | 13.0 us | 5.0 ms | 0.67 ms, **13.4 %** |
| qwen3.6-35B MoE | 1298 | 9.3 us | 14.2 ms | 2.31 ms, **16.3 %** |
| qwen3.8-27B | 1745 | 39.0 us | 66 ms | 3.11 ms, **4.7 %** |

The 27B issues more dispatches than the MoE and cares least, because its kernels are four times longer.
The MoE has the shortest kernels of the three and is the most exposed.

That last column was an assumption when this was written, at a round one microsecond a dispatch. It has
since been measured on this board and it is 1.78 microseconds by a fit, or 2.03 for an empty kernel
([`logs/dispatch-floor-2026-09-22/`](../dispatch-floor-2026-09-22/)), so the column is recomputed above
and the MoE's share of its token comes out at 16.3 percent against a deficit of 17.2.

That coincidence was read as the answer to the question this page left open, and it is not.
[`logs/floor-vs-vulkan-2026-09-23/`](../floor-vs-vulkan-2026-09-23/) measured Vulkan's floor, which is
the **larger** of the two, and counted Vulkan's dispatches, which are 1423 a token against ROCm's 1298.
ROCm dispatches less often and more cheaply than the backend that beats it. The 16.3 percent above also
cannot be charged on top of the kernels: the trace on this page puts 775.832 ms of kernel time in 64
tokens, 12.12 ms of a 13.93 ms token, so only 1.81 ms of the token is outside a kernel at all. **The
deficit is not the dispatch floor**, and where it is remains open.

## HIP graphs do not close that gap here

The same trace with `GGML_CUDA_DISABLE_GRAPHS=1` ([`trace-moe-tg-graphs-off.txt`](trace-moe-tg-graphs-off.txt)):
the same 83060 dispatches, and the 2 to 5 us gap bin holds 199.2 ms with graphs on against 203.9 ms with
them off. **Graph replay is not reducing the per-dispatch gap on this board at all.** Whatever that gap
is, it is not host-side launch cost, because removing host-side launch cost is what graph replay does.

Whether that costs anything end to end is a smaller and messier question, and three measurements of it
disagree, so no setting is recommended here. An interleaved A/B of three rounds read graphs-off 2.8
percent ahead on the 1.5B, 1.8 on the MoE and 0.5 on the 27B. A second A/B of five rounds, with the arm
order reversed on alternate rounds, read 1.9 and 2.2 percent, but with the 1.5B's rounds split three to
two, not five to nothing:

| round | 1.5B off / on | MoE off / on |
|---|---|---|
| 1 | 1.023 | 1.018 |
| 2 | 0.986 | 1.014 |
| 3 | 1.030 | 1.045 |
| 4 | 0.991 | 1.016 |
| 5 | 1.021 | 1.022 |

And a whole campaign with graphs disabled ([`campaign-nograph/`](campaign-nograph/)) reads every model
within 0.4 percent of the campaign with them on
([`logs/fedora44-campaign-iq-shmem-2026-09-20/`](../fedora44-campaign-iq-shmem-2026-09-20/)): 1.004 on
the 1.5B, 1.002 on the MoE, 1.004 on the 27B, and 0.996 to 0.997 on the three that go the other way.

The campaign puts a Vulkan run of the same model between consecutive ROCm runs and the A/Bs run them
back to back, which is the only difference between the two arrangements that is visible from here. It is
enough to move the ROCm figure by more than the effect under test, so the effect is not established: on
the MoE all five rounds of the longer A/B favour graphs off, which is suggestive, and nothing else is.
The trace finding above stands on its own and does not depend on it.

## Prefill is more than half MMQ on the experts

The same tool on `-p 512 -n 0` ([`trace-moe-pp512.txt`](trace-moe-pp512.txt)). Prefill traces honestly
where decode does not: `llama-bench` reported 339.06 t/s while being traced against 340.4 in the
campaign, because the kernels are large and few. 4006 dispatches, 3040 ms of kernel time:

| dispatches | ms | share | avg us | kernel |
|---|---|---|---|---|
| 160 | 1144.5 | 37.6 % | 7152.8 | MMQ, iq2_xxs, expert path |
| 360 | 790.1 | 26.0 % | 2194.7 | packed-fp16 GEMM, q5_K |
| 74 | 536.1 | 17.6 % | 7244.0 | MMQ, iq3_xxs, expert path |
| 140 | 177.7 | 5.8 % | 1269.0 | packed-fp16 GEMM, q6_K |
| 60 | 104.5 | 3.4 % | 1740.9 | `gated_delta_net_cuda` |
| 200 | 61.6 | 2.0 % | 308.2 | rocBLAS SGEMM (Tensile) |
| 6 | 34.7 | 1.1 % | 5781.8 | MMQ, iq4_xs, expert path |

**MMQ on the expert path is 56.3 percent of this model's prefill**, because patch 8's GEMM hooks
`ggml_cuda_mul_mat` and the experts arrive through `ggml_cuda_mul_mat_id`. That is why patch 8 was worth
only 8 percent here against 47 to 51 on the dense 14B models.

Extending the GEMM to that path is the largest single item left in the project, and it is worth less
than it first looks, for two reasons. The shape is wrong for the tile: `ffn_moe_gate` is
`iq2_xxs[2048,512,256]`, 256 experts of 512 rows, and at pp512 each expert sees about 16 of the 512
tokens (512 tokens times 8 experts used, spread over 256 experts), 64 at pp2048. Patch 8's kernel needs
256 tokens before a 128-wide column tile pays, which is exactly the predicate it ships with, so the
expert path would need a thin-column variant instead of the existing one. And the gap to close is
smaller here than elsewhere: replaying the pp2048 graph, ROCm's MMQ is 1.36 times Vulkan on
`ffn_moe_gate` and 1.21 to 1.23 on the two `ffn_moe_down` shapes, where the dense shapes on the same
build were 1.8 to 2.1. Closing the expert gap entirely would be worth about 11 percent of this model's
prefill, taking 340 t/s to roughly 383 against Vulkan's 457; the rest of the deficit is spread over the
shapes the GEMM already covers.

## Files

`trace-moe-tg-graphs-{on,off}.txt` and their `.bench` are the two decode traces and what `llama-bench`
reported during each, `trace-moe-pp512.*` the prefill one. `ops-moe-n1.txt` is the exported one-token graph, `ops-moe-n1-{hip,vk}.log` the two
replays and `opcmp-moe-n1.txt` their comparison. `sdma0-*.jsonl` and `nosdma-*.jsonl` are the
three-round graphs on/off A/B with and without `HSA_ENABLE_SDMA=0`, `ab5-*.jsonl` the five-round one with
the arm order alternating, and `campaign-nograph/` the whole campaign with graphs disabled.
[`scripts/graphs_ab.sh`](../../scripts/graphs_ab.sh) is the A/B script.
