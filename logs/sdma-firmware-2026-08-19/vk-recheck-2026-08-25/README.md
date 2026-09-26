# Vulkan re-measurement, 25 August

The parent page quotes three Vulkan figures for the navi12 microcode state that
were transcribed from a run nobody kept. This is that measurement taken again,
six days later, on a board still carrying the substituted firmware.

Same invocation the campaign uses, `llama-bench -mmp 0 -ngl 99 -fa on -p 512
-n 64 -r 2`, three repeats per model, on the Vulkan build.

| model | pp512 | tg64 |
|---|---|---|
| qwen2.5-1.5B Q4_K_M | 1842.21 to 1843.42 | 210.71 to 211.11 |
| qwen3-8B Q8_0 | 400.75 to 400.81 | 39.05 to 39.11 |

The parent's 1843.11, 210.75 and 39.11 all sit inside these spreads. That does
not recover the original numbers, and a reproduction six days later is weaker
evidence than the run itself would have been, but the claim it supports, that
the firmware substitution leaves Vulkan unchanged, survives being checked.

`log` records the configuration and the per-repeat figures; the `vk_*.log` files
are the raw `llama-bench` output. Zero faults during the run.
