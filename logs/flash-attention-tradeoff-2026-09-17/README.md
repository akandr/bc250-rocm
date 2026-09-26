# Flash attention is a trade at depth, not a default, 2026-09-17

Everything in this repository is measured with `-fa on`, and the manual says to use it. Measuring prefill at
context depth for the first time showed that is wrong for half the cases.

Fedora 44, kernel 7.2.5 with the bc250 module, clock pinned at 1500 MHz, qwen3-8B Q8_0, one test per
`llama-bench` invocation, `-r 2`, both backends from the same llama.cpp tree.

## Prefill: faster with flash attention off

`-p 2048 -n 0` at the given existing context depth, tokens per second:

| depth | ROCm, fa on | ROCm, fa off | Vulkan, fa on | Vulkan, fa off |
|---|---|---|---|---|
| 0 | 198.2 | 250.4 | 367.5 | 374.8 |
| 4096 | 98.1 | 186.1 | 216.9 | 255.3 |
| 8192 | 65.5 | 133.3 | 144.0 | 220.4 |

Turning flash attention off doubles ROCm prefill at 8192 tokens of context and adds half again on Vulkan.

## Decode: much faster with it on

`-p 0 -n 64` at the same depths:

| depth | ROCm, fa on | ROCm, fa off | Vulkan, fa on | Vulkan, fa off |
|---|---|---|---|---|
| 4096 | 32.7 | 18.6 | 35.7 | 30.2 |
| 8192 | 28.3 | 12.5 | 33.2 | 23.7 |

Here the sign reverses and the margin is larger: on ROCm at 8192, flash attention is worth 2.3 times.

## The rule

- Long prompt, short answer, and repeated prefill of a growing context: `-fa off` is much faster.
- Generation-heavy work at depth: `-fa on`, and on ROCm it is not close.
- Flash attention also keeps the KV cache smaller, so turning it off costs usable context. The ceilings in
  [`../fedora44-ceilings-2026-09-16/`](../fedora44-ceilings-2026-09-16/) were measured with it on.

## Correctness

The ROCm gates with `-fa off` read 8.9597 on the 1.5B and 9.1448 on the 8B, against 8.9442 and 9.1117 with
it on. The paths accumulate in a different order, so the values differ in the third decimal; both are
correct answers, and the fp16 defect this repository chased produced errors orders of magnitude larger.
Anyone gating on an exact value has to gate per flag.

## Also measured here

llama.cpp master `bfdc321` with the two remaining patches prefills 1.5 to 3 percent slower than the measured
base at every depth on ROCm (195.2, 95.7 and 63.4 against 198.2, 98.1 and 65.5), consistent with the small
regression seen at depth 0 earlier.
