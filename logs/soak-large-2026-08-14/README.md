# Large-model endurance soak, 2026-08-14

8.1 hours rotating three large models, 78 rounds, 26 per model. Each round runs
a prefill benchmark, an eight-chunk perplexity gate, and once per rotation the
allocation-churn sweep.

`soak.log` holds the per-round results. `thermals.tsv` holds the temperature
and clock samples.

The result is one distinct perplexity value per model across all 26 of its
rounds: 7.3503 for the 8B, 6.3970 for the 14B, 5.1887 for the 35B MoE. Prefill
holds to within a percent, all 26 churn sweeps complete with no faults, and the
temperature peaks at 94C against a 77C mean.

Because each round loads a multi-gigabyte model from scratch, this also
exercises the large-model load path, which used to be the least reliable part of
this system. The 8B value was re-measured on a fresh boot three days later and
returned 7.3503 again.

Produced by the large-model soak harness, rotating three models with a perplexity gate each round; the same shape as `scripts/soak_current_stack.sh`, which superseded it.
