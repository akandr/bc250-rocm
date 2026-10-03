# With `GGML_CUDA_GRAPH_OPT=1` the Q branch overwrites `attn_norm` while K and V still read it, 2026-10-02

## What was open

The main README recommended `GGML_CUDA_GRAPH_OPT=1` for decode speed
([`logs/graph-opt-2026-09-24/`](../graph-opt-2026-09-24/)). Its correctness was checked with the two
perplexity gates, and those cannot see what it does: perplexity evaluates prompts, and the option only
acts on single-token decode, where ggml-cuda captures HIP graphs. Generated text was never compared.

## Result

128 greedy tokens from the same prompt (`llama-cli -st -c 4096 --temp 0 -s 1 -n 128`) on the campaign
build, `build-hip-pkf16`, every reply hashed, with every other GPU user stopped (`repeat.log`,
`repeat-replies/`):

| model | default | graphs disabled | `GGML_CUDA_GRAPH_OPT=1` |
|---|---|---|---|
| qwen3-14B Q4_K_M | 3 of 3 identical, `679a50d9` | 2 of 2 identical to the default | 3 of 3 different, all word salad |
| qwen3-8B Q8_0 | 3 of 3 identical, `90b056d9` | 2 of 2 identical to the default | 3 of 3 different, all word salad |

The default reply begins *Okay, so I need to explain what a GPU compute queue is and why a driver bug
in it matters*; the option's begins *its     ,      亚>斯:8》 (),, -, (,5',5 (5),5)\*\*5)ﬃ,s,,  用*, and
the next run's *its0  0   a any isam  and that , the)-)) and and or players)*.

## Where: the Q branch writes into `attn_norm`

`scripts/graphopt_mechanism.sh` builds a copy of the campaign's source tree with llama.cpp PR #27301
applied, which lets a backend's `graph_optimize` add allocation dependencies, and with two switches
added by `scripts/graphopt_instrument.py` (`instrumented.diff`):

- `GGML_CUDA_GRAPH_OPT_DEBUG=1` checks every tensor a concurrent region writes against every tensor
  that another stream of the same region reads from outside it, and dumps the first region;
- `GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1` keeps every tensor of a region, and every tensor it reads,
  allocated until the region's join.

Layer 0 of qwen3-8B without the dependencies, abridged from `region-layer0.txt`:

    stream 1 MUL_MAT    Qcur-0          [0x7fbced608400, +16384) <- attn_norm-0 [0x7fbced604400, +16384)
    stream 1 RMS_NORM   norm-0          [0x7fbced604400, +16384)
    stream 1 MUL        Qcur_normed-0   [0x7fbced604400, +16384)
    stream 1 ROPE       Qcur-0          [0x7fbced604400, +16384)
    stream 2 MUL_MAT    Vcur-0          [0x7fbced60c400, +4096)  <- attn_norm-0 [0x7fbced604400, +16384)
    stream 3 MUL_MAT    Kcur-0          [0x7fbced60d400, +4096)  <- attn_norm-0 [0x7fbced604400, +16384)

The allocator plans memory for sequential execution. Once the last of the three projections has been
issued, `attn_norm`'s buffer is free in that order, and the Q branch's output is placed in it. On three
streams the Q branch can write there while the K and V projections on streams 3 and 2 are still
reading it. On qwen3 the writer is the fused kernel for the RMS norm, its weight and the rotation; on
the two qwen2 models it is the rotation after the bias add. The check finds this in every region of
every model on which the option launches streams, two overlaps a region, the same two each time
(`mechanism-debug/`):

    graphopt-recycle: capture 0: Qcur-0 (ROPE, stream 1) writes [0x7fbced604400, +16384) over attn_norm-0 (MUL), which Kcur-0 (MUL_MAT, stream 3) reads at [0x7fbced604400, +16384)
    graphopt-recycle: capture 0: Qcur-0 (ROPE, stream 1) writes [0x7fbced604400, +16384) over attn_norm-0 (MUL), which Vcur-0 (MUL_MAT, stream 2) reads at [0x7fbced604400, +16384)

| model | regions a token | regions with the overlap | overlaps |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 28 | 28 | 56 |
| qwen3-8B Q8_0 | 36 | 36 | 72 |
| qwen3-14B Q4_K_M | 40 | 40 | 80 |
| deepseek-r1-14B Q4_K_M | 48 | 48 | 96 |

`ggml_cuda_concurrent_event::is_valid()` does not see it. It compares the branches' outputs with each
other and rejects a branch that reads another branch's output, but `attn_norm` is the fork node, outside
every branch. The MoE and the 27B are not affected only because every one of their regions is refused
before it can run ([`logs/moe-stream-aliasing-2026-09-25/`](../moe-stream-aliasing-2026-09-25/)).

## What it does depends on how the streams interleave

The instrumented build, the same prompt (`mechanism.log`, `mechanism-replies/`):

| model | default | `GRAPH_OPT=1` | `GRAPH_OPT=1` with the dependencies |
|---|---|---|---|
| qwen3-8B Q8_0 | `90b056d9` | 2 runs, 2 different word salads | 3 of 3 `90b056d9` |
| qwen3-14B Q4_K_M | `679a50d9` | word salad | 3 of 3 `679a50d9` |
| qwen2.5-1.5B Q4_K_M | `7033302a` | 23 bytes of garbage | `7033302a` |
| deepseek-r1-14B Q4_K_M | 2 of 2 `96357da6` | 2 runs, 2 different coherent replies, neither the default | 2 of 2 `96357da6` |

The 1.5B is the one to remember. On the campaign build its replies with and without the option were
byte-identical in three ordinary runs (`first-check.log`); on this build the same option turned it into
garbage. A race whose effect depends on timing is not cleared by a run that comes out right.

## The dependencies fix it and keep the speed

`alloc-deps.diff` is PR #27301 for the backends plus the dependency loop in
`ggml_backend_cuda_graph_optimize`, behind `GGML_CUDA_GRAPH_OPT_ALLOC_DEPS=1` as it was tested: the
instrumented build ran the same code with the diagnostics compiled in and switched off. With it every
region still runs on three streams, and the overlap is gone:

    graphopt-debug: capture 0: 36 regions, all valid; 36 ran on concurrent streams, 0 of them writing over 0 sources another stream reads

The Q branch gets its own buffer (`Qcur-0` at `0x7f5251018400`, `attn_norm-0` at `0x7f5251004400`), and
decode keeps the option's whole gain (`bench/`, `llama-bench -mmp 0 -ngl 99 -fa on -p 0 -n 64 -r 3`, two
rounds, median of the second and third repetitions, tokens per second):

| model | default | option, as shipped | option with the dependencies |
|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 198.09 | 213.32 (1.077) | 213.11 (1.076) |
| qwen3-8B Q8_0 | 38.52 | 39.72 (1.031) | 39.74 (1.032) |
| deepseek-r1-14B Q4_K_M | 32.33 | 33.34 (1.031) | 33.33 (1.031) |
| qwen3-14B Q4_K_M | 32.33 | 33.35 (1.032) | 33.36 (1.032) |

The fixed option is within 0.1 percent of the unfixed one on every model. The first repetition of every
arm is slower than the other two, 187.5 and 194.2 against 198 for the 1.5B's default, which is why it is
left out; counting it changes no ratio by more than 0.003.

## Upstream

The region code, the Q/K/V part of `ggml_backend_cuda_graph_optimize` and `is_valid()`, is identical
in upstream master at the time of writing. Master uses the allocation-dependency hook for its MoE
fusions, not for these regions. The allocator's decision does not depend on the backend, so the same
overlap should be there on CUDA; whether it corrupts output there depends on how the GPU schedules the
three streams, which nothing here measures.

## Caveats

- The instrumented copy reproduces the campaign build's default replies byte for byte on qwen3-8B and
  qwen3-14B, but not on the 1.5B: `431965dd` there, `7033302a` here, both coherent and each
  reproducible. Some difference between the source tree it was copied from and the one
  `build-hip-pkf16` was built from on 21 September has not been found. Every comparison in the tables
  above is within one build.
- `first-check.log` is the first, rougher pass. An `ollama` service, restarted by the previous
  measurement's cleanup, was running for its first twenty minutes; its 8B default run found no ROCm
  device and fell back to the CPU until the timeout; and the Vulkan build has no `llama-cli`. It is kept
  because it is where the problem first showed. The tables come from the two later passes, with every
  other GPU user stopped.
- The check ran in a pause of a long ceiling campaign, frozen between two of its runs.
- CUDA is not tested.

## Files

- `repeat.log`, `repeat-replies/`: the campaign build, three replies per configuration
  (`scripts/graphopt_repeat.sh`).
- `first-check.log`: the first pass (`scripts/graphopt_check.sh`).
- `mechanism.log`, `mechanism-replies/`, `mechanism-debug/`, `bench/`: the instrumented build
  (`scripts/graphopt_mechanism.sh`), with each run's diagnostics and each benchmark's samples.
- `region-layer0.txt`: the first region of qwen3-8B with and without the dependencies, and the order
  the streams ran it in.
- `instrumented.diff`: everything the instrumented build adds to the source tree.
- `alloc-deps.diff`: the fix alone.

## How to reproduce

`scripts/graphopt_repeat.sh` on any build with `llama-cli` gives three greedy replies per
configuration; compare the hashes. `scripts/graphopt_mechanism.sh` builds the instrumented copy, which
needs PR #27301's diff (`gh pr diff 27301 --repo ggml-org/llama.cpp`) and
`scripts/graphopt_instrument.py`, and runs the tables above.
