# Perplexity gates against Vulkan, 2026-08-14

Every inference rate quoted in the top-level README is gated: before a
tokens-per-second number is recorded, the same model and text are run through
`llama-perplexity` on both backends and the values compared. A backend that
produces fast nonsense looks identical in a rate column and is caught here.

`gates.log` is the summary; the per-model `*_hip.log` and `*_vk.log` files are
the runs behind it. Eight chunks each.

| model | ROCm | Vulkan | difference |
|---|---|---|---|
| qwen3-8B Q8_0 | 7.3503 | 7.3792 | 0.4 percent |
| qwen3-14B Q4_K_M | 6.3970 | 6.4548 | 0.9 percent |
| deepseek-r1-14B Q4_K_M | 6.0013 | 6.0416 | 0.7 percent |
| qwen3.6-35B-A3B MoE IQ2_M | 5.1887 | 5.2041 | 0.3 percent |

The ROCm value is slightly below the Vulkan one on all four, consistently
rather than randomly, which is what the top-level README notes when discussing
how far agreement can be pushed.

The 8B value here, 7.3503, is the same one the eight-hour large-model soak
returned on all 26 of its rounds, and re-measuring it on a fresh boot in
August returned 7.3503 again.

Collected by the perplexity arms of `scripts/bench_fixed_stack.sh`; each gate is one `llama-perplexity` invocation on the model and command named in the table.
