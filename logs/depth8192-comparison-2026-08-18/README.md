Ten decode runs each of three models at a primed depth of 8192, chosen because
every model keeps HIP graph capture on at that depth. An earlier comparison at
depth 16128 was unusable: the 14B requires `GGML_CUDA_DISABLE_GRAPHS=1` past a
primed depth of 12000 and the other two models did not use it, and that flag has
a large effect of its own.

| model | share of the 402 GiB/s ceiling | mean t/s | sd | coefficient of variation |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K | 29 percent | 87.28 | 2.41 | 2.8 percent |
| qwen3-14B Q4_K | 42 percent | 11.40 | 0.58 | 5.1 percent |
| qwen3-8B Q8_0 | 80 percent | 21.93 | 1.44 | 6.6 percent |

Read with care. This run still presented the models in blocks, all ten of one
before any of the next, so anything drifting over the hour it took is not
separated from the model. The counterbalanced version, with model order rotated
each round, is in `../counterbalanced-2026-08-18/` and is what the write-up
draws on. This directory is kept because it is the run the flush-cost
measurement shares a harness with.
