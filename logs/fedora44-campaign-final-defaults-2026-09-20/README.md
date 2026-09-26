# Confirmation campaign on the shipped defaults, 2026-09-20

The kernel gained two run-time switches while it was being tuned, and both ended up off or at their
starting value: the f32 promotion stays at every stage
([`logs/round13-2026-09-20/`](../round13-2026-09-20/)), tile prefetching is off
([`logs/round14-2026-09-20/`](../round14-2026-09-20/)), and q8_0 is excluded
([`logs/round11-2026-09-20/`](../round11-2026-09-20/)). This campaign is the same script and models on a
build with exactly those defaults, to confirm that the front page's numbers describe what the patch
actually ships.

| model | ROCm pp512 | Vulkan pp512 | ROCm tg64 | Vulkan tg64 |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1250.3 | 1849.6 | 198.1 | 212.4 |
| qwen3-8B Q8_0 | 279.4 | 394.7 | 38.5 | 39.0 |
| deepseek-r1-14B Q4_K_M | 145.1 | 199.7 | 32.3 | 34.9 |
| qwen3-14B Q4_K_M | 151.0 | 204.5 | 32.3 | 34.6 |
| qwen3.6-35B-A3B MoE IQ2_M | 340.3 | 457.9 | 70.7 | 86.6 |
| qwen3.8-27B UD-IQ3_XXS | 81.2 | 105.0 | 14.8 | 17.6 |

Every row reproduces [`logs/fedora44-campaign-pkf16-all-2026-09-20/`](../fedora44-campaign-pkf16-all-2026-09-20/)
within 0.3 percent, so the front page stands as written.

`round15-log` also contains a quick check taken immediately after the build that reads the 14B at 125
and the 1.5B at 1224, both low. Its MMQ control is low by the same margin (96.3 against the campaign's
99, 905 against 915), which is what a warm board looks like; the campaign that follows it, nine samples
per model alternating with Vulkan, is the measurement.
