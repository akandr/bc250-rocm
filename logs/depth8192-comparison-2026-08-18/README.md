Ten decode runs each of three models at a primed depth of 8192, chosen because
every model keeps HIP graph capture on at that depth. An earlier comparison at
depth 16128 was unusable: the 14B requires `GGML_CUDA_DISABLE_GRAPHS=1` past a
primed depth of 12000 and the other two models did not use it, and that flag has
a large effect of its own.

| model | share of the 402 GiB/s ceiling | mean t/s | sd | coefficient of variation |
|---|---|---|---|---|
| qwen2.5-1.5B Q4_K | 29 percent | 87.28 | 2.41 | 2.8 percent |
| qwen3-14B Q4_K | 46 percent | 11.40 | 0.58 | 5.1 percent |
| qwen3-8B Q8_0 | 80 percent | 21.93 | 1.44 | 6.6 percent |

One note on the bandwidth-share column, because it is easy to misread and it
misled the author of this note on a later reading. It is not computed from the
decode rates in this table. It is the model-level figure from the companion, the
file size times the model's ordinary decode rate against the 402 GiB/s ceiling,
which the companion's own table states and which reproduces exactly: 8.24 GiB at
39.2 t/s is 323 GiB/s or 80 percent, 8.63 at 21.5 is 186 or 46 percent, 1.04 at
113.5 is 118 or 29 percent.

The middle row previously read 42 percent, which is deepseek-r1-14B's share, not
this model's. The bandwidth table it was taken from lists only one 14B and that
one is the deepseek distill, 8.37 GiB decoding at 20.3 t/s, while the runs here
used qwen3-14B, 8.63 GiB at 21.5, as every `p1_q14_*.log` header records. The
ordering the comparison turns on is unchanged, since 46 percent still sits
between 29 and 80. Dividing the shares by the depth-8192 rates in the
column beside them instead gives implausible per-token figures and suggests the
derivation is missing. It is not missing; it is a property of the model rather
than of this run.

Read with care. This run still presented the models in blocks, all ten of one
before any of the next, so anything drifting over the hour it took is not
separated from the model. The counterbalanced version, with model order rotated
each round, is in `../counterbalanced-2026-08-18/` and is what the write-up
draws on. This directory is kept because it is the run the flush-cost
measurement shares a harness with.
