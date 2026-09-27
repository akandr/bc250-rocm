# Where the ROCm prefill deficit lives, operation by operation, 2026-09-17

[`scripts/op_perf_hip_vs_vulkan.sh`](../../scripts/op_perf_hip_vs_vulkan.sh) on the default boot:
Fedora 44, kernel 7.2.5 with the bc250 amdgpu module, native gfx1013 rocBLAS 7.1.1 and the corrected
comgr from `/opt/bc250-rocm`, GPU clock policy at 1500 MHz with `oberon-governor` active, `ollama`
stopped.

ROCm prefilled far behind Vulkan at this point, the qwen3-8B at 198.2 tokens/s against 367.5 at
pp2048 with flash attention on ([`../flash-attention-tradeoff-2026-09-17/`](../flash-attention-tradeoff-2026-09-17/)),
and until now nothing said which operation carried that. The replay below uses the qwen2.5-1.5B's
graph. `test-export-graph-ops` writes out the real graph of a pp2048 prefill (no
weights needed), and `test-backend-ops perf --test-file` replays exactly those shapes on each
backend, so the two are compared on what the model runs, not on a synthetic grid. Both
backends run inside one pass and the pass is repeated. Exported twice, once with `-fa off` and once
with `-fa on`, because the two configurations run different graphs.

## Result with `-fa off`: the deficit is the matmuls, and only the matmuls

| prefill ops (batch 2048) | cases | median HIP/Vulkan time |
|---|---|---|
| `MUL_MAT` | 11 | **1.84** |
| everything else (`ADD`, `MUL`, `RMS_NORM`, `SOFT_MAX`, `SWIGLU`, `ROPE`, `GET_ROWS`, `SET_ROWS`, `CONT`) | 11 | **0.75** |

Summed over the prefill shapes, excluding the output head (llama.cpp computes logits only for the
last token, so the graph's full-batch `[151936,2048]` case is not work a real prefill does), HIP
spends 89.2 ms against Vulkan's 51.2 ms, a ratio of **1.74**, so within the graph the whole deficit is
in the matmuls. An earlier revision set this against an end-to-end ratio of 367/197 = 1.86 and called
the two in agreement, but those are the qwen3-8B's figures with `-fa on`, not this graph's, and no
end-to-end `-fa off` pair for the 1.5B was taken that day. The op-level result stands without an
end-to-end cross-check (corrected 27 September).

Both the quantized path and the f16 attention path are slow, so this is not one kernel family.

| shape | HIP us | Vulkan us | ratio |
|---|---|---|---|
| `q6_K[8960,1536] x f32[8960,2048]` ffn_down | 22688 | 11535 | 1.97 |
| `q4_K[8960,1536] x f32[8960,2048]` ffn_down | 20654 | 10385 | 1.99 |
| `q4_K[1536,8960] x f32[1536,2048]` ffn_up/gate | 19503 | 11704 | 1.67 |
| `f16[128,4096,2] x f32[4096,2048,12]` KQV | 8208 | 5805 | 1.41 |
| `f16[4096,128,2] x f32[4096,2048,12]` KQ | 8087 | 4399 | 1.84 |
| `q4_K[1536,1536] x f32[1536,2048]` attn qkv/o | 3532 | 1790 | 1.97 |
| `q6_K[1536,256] x f32[1536,2048]` | 824 | 455 | 1.81 |
| `q4_K[1536,256] x f32[1536,2048]` | 756 | 413 | 1.83 |

The ROCm side wins on the memory-bound elementwise work, by a wide margin on the broadcast shapes:
`ADD[1536,2048] + f32[1536,1]` 92.9 against 210.9 us and the matching `MUL` 92.8 against 210.7, both
0.44; `ADD[256,2048]` 0.49; `GET_ROWS` and the large `ADD` 0.75. `SWIGLU`, `SET_ROWS` and the large
`SOFT_MAX` are within a few percent either way. `ROPE` (1.64) and `RMS_NORM` (1.29) go the other way,
but they are 199 and 126 us against matmuls of 20000.

## Result with `-fa on`: the flash-attention kernel is seven times slower

This is the larger finding, and it is specific to batched flash attention.

| shape | HIP us | Vulkan us | ratio |
|---|---|---|---|
| `FLASH_ATTN_EXT` `ne=[128,12,2048,1]` (prefill) | 114650 | 16260 | **7.05** |
| `FLASH_ATTN_EXT` `ne=[128,12,1,1]` (decode) | 46.6 | 44.6 | 1.05 |

At pp2048 the flash-attention kernel becomes the single largest cost in the ROCm prefill graph after
the output head, at 114.7 ms against every matmul's 20 ms or less. Vulkan's costs 16.3 ms. The
decode-shaped case is at parity, so this is the batched kernel and not flash attention generally.

It is also worse than ROCm's own non-flash path: the KQ and KQV matmuls it replaces total 16.3 ms
with `-fa off`, so turning flash attention on costs ROCm seven times what it saves. On Vulkan the
same switch goes from 10.2 ms to 16.3 ms, a mild loss. That is the kernel-level cause of the trade
this repository already measured end to end in
[`logs/flash-attention-tradeoff-2026-09-17/`](../flash-attention-tradeoff-2026-09-17/), where ROCm
prefill is much faster with `-fa off` while decode is much faster with `-fa on`. The decode side of
that trade is not explained by these numbers, since decode-shaped flash attention is at parity here.

The `-fa on` matmul picture is unchanged: median 1.80 over 7 cases, 0.75 for the rest.

### The deficit is flat in batch size, so it is not an occupancy cliff

If the RDNA tile configuration were wrong at large `ncols`, the cost per token would climb
with the micro-batch. It does not. One graph export per micro-batch, same KV length of 4096
(`fa-batch-scan.log`):

| micro-batch | HIP us | Vulkan us | ratio | HIP us/token | Vulkan us/token |
|---|---|---|---|---|---|
| 128 | 7770 | 1308 | 5.94 | 60.7 | 10.2 |
| 512 | 30902 | 4138 | 7.47 | 60.4 | 8.1 |
| 1024 | 57807 | 8317 | 6.95 | 56.5 | 8.1 |
| 2048 | 114664 | 16249 | 7.06 | 56.0 | 7.9 |

Both backends scale linearly and ROCm's cost per token is flat, if anything falling slightly with
batch. The kernel is uniformly about seven times slower instead of falling off a cliff at some
tile size.

### And it is not the wrong tile table either

gfx1013 falls to the generic tile kernel, since it has none of MFMA, WMMA or Turing MMA, and the
`RDNA1` macro routes it to `ggml_cuda_fattn_tile_get_config_amd_rdna`, a table whose entries were
tuned on RDNA2 and later. Tested directly by building the same tree with RDNA1 sent to the generic
AMD table instead, on both the host and the device side (`fa-tile-config-ab.log`):

| build | pass 1 | pass 2 |
|---|---|---|
| RDNA table (production) | 114674 us | 114533 us |
| generic AMD table | 111329 us | 111267 us |

**2.9 percent, reproducible, and nowhere near seven times.** The tile table is not the cause, so this
is not a configuration that can be corrected by routing gfx1013 elsewhere in the existing tables.
The source change was reverted. What does make the HIP tile kernel slow here is still open.

## Stability

`-fa off`, three passes: ratios reproduce to two decimals on every large case. The three ffn matmuls
read 1.99/1.99/1.99, 1.97/1.97/1.97 and 1.97/1.97/1.97, and the KQ matmul 1.84/1.84/1.84. The widest
pass-to-pass spread on any case is 1.08, on `CONT`, a 378 us case sitting next to 20000 us ones.

`-fa on`, two passes: the prefill flash-attention ratio is 7.07 and 7.03.

The governor logged one overheat at 17:06, before these passes. Both backends run inside each pass
and the ratios do not move, so thermal drift is not distorting them.

## Files

| | |
|---|---|
| `ops-1.5b-pp2048.txt` | exported graph, `-fa off`, 45 shapes |
| `ops-fa-on.txt` | exported graph, `-fa on`, 37 shapes |
| `hip-graph-p1.log` .. `p3`, `vk-graph-p1.log` .. `p3` | `-fa off` passes |
| `hip-faon-p1.log` .. `p2`, `vk-faon-p1.log` .. `p2` | `-fa on` passes |
| `fa-batch-scan.log` | flash attention against micro-batch, both backends |
| `fa-tile-config-ab.log`, `f44-faon-x*.log`, `fatile-faon-x*.log` | the tile-table A/B |

## What this does not say

It does not say why the kernels are slower, only which ones are. The quantized cases are served by
each backend's own quantized matmul kernels, MMQ on the HIP side and compute shaders on the Vulkan
side, so that half is a kernel-quality difference between the two backends and not anything
about rocBLAS. The f16 cases are the ones where the untuned native Tensile build could matter, and
they were separated on 18 September with [`hgemm_attn.cpp`](hgemm_attn.cpp), which issues the same
strided-batched fp16 GEMM the KQ op needs, twelve heads at m=4096, n=2048, k=128, straight to rocBLAS
(`hgemm-attn.log`). With f32 accumulation, which is what the KQV-precision patch asks for, the library
call alone takes 7.29 ms of the 8.09 ms op: rocBLAS is 90 percent of the KQ time, at 3.5 TFLOP/s,
against Vulkan's 4.40 ms for the whole op. With f16 accumulation it is 5.54 ms. So the f16 half of the
`-fa off` gap is Tensile throughput at these shapes, not llama.cpp's surrounding work. The KQV shape
does not compare cleanly: issued plainly it takes 8.63 ms, longer than llama.cpp's whole 8.21 ms op, so
llama.cpp issues that one in a different layout. Since the flash-attention fix made `-fa on` the faster
prefill setting, this path matters less than it did when these numbers were first taken.

The flash-attention half was resolved the same evening: the kernel spills 569 registers on RDNA1
and a retuned tile configuration recovers most of it
([`logs/rdna1-fattn-spill-2026-09-17/`](../rdna1-fattn-spill-2026-09-17/)). The same register
check was then pointed at the quantized matmul kernels, and they do not spill: MMQ sits at 242
VGPRs on gfx1013 against 233 on gfx1030 with no scratch on either, and MMVQ at 30 to 125 with
occupancy 8 to 20. Vulkan on this board has no hardware help either, since RADV reports every
`integerDotProduct*Accelerated` property false and `llama-bench` prints `matrix cores: none`, `int
dot: 0`; its q4_K matvec shader unpacks nibbles and runs float `fma` chains. So the 1.7 to 2.0x on
quantized matmuls is a difference of kernel formulation between the two backends, HIP's emulated
int8 dot product against Vulkan's float arithmetic, and not something a configuration table fixes.
