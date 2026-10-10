# The earlier passes, with llama-bench's copy

Two campaigns before this one measured the same ceilings with llama-bench as shipped, which copies the
sequence state into host memory after every depth fill. Their passes seed this campaign as lower bounds
(`../seed-ok.tsv`); their failures do not carry over.

- `compare-summary.tsv`: [`scripts/ceilings_compare.sh`](../../../scripts/ceilings_compare.sh), 27 September,
  ROCm and Vulkan, a doubling ladder with a depth cap per model.
- `bisect-summary.tsv`, `bisect-results.txt`: [`scripts/ceilings_bisect.sh`](../../../scripts/ceilings_bisect.sh)
  with the stock llama-bench, 28 to 30 September, all four arms. It was stopped once the copy was found,
  before the 1.5B, and with the MoE's `q4_0` cell measured on two arms only.
- `runs/`: llama-bench's output for every earlier pass that is a cell's deepest pass in `../ceilings.tsv`.
