Are the three llama.cpp patches still needed, and do they still apply?

Everything in this repository is measured at master 7ba604f (2026-08-09). The
write-up previously claimed the patches were "still needed against current
master", which is a claim about a moving target that nothing was checking. This
is the check, and it should be repeated rather than trusted.

Checked 2026-08-19 against upstream master ee4c505, 174 commits later, in a
detached worktree so the measurement tree was untouched:

| patch | site upstream | applies |
|---|---|---|
| 0001 `prop.integrated` | `ggml-cuda.cu:305` still `= prop.integrated` for HIP | cleanly |
| 0002 KQV precision | `llama-graph.cpp:2627` still has no `set_prec` on kqv | cleanly |
| 0003 gfx1013 RDNA1 macro | `vendors/hip.h:232` still lists only gfx1010 and gfx1012 | cleanly |

All three sites are unchanged, so all three are still needed. Line numbers have
moved by a few lines in each file, which the patches absorb.

Note the non-HIP branch beside the `prop.integrated` line still carries
upstream's own comment about corrupted output on integrated GPUs, which is the
precedent that patch 0001 applies to the HIP branch.

Reproduce with `git fetch origin master`, `git worktree add --detach <dir>
FETCH_HEAD`, then `git apply --check` each patch.
