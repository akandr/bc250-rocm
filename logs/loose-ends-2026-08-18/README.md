# Closing the remaining loose ends, 2026-08-18

Four things this repository left open or under-explained. Each is investigated
here rather than left as a caveat.

## 1. The 8B decode variance

The 8B at a fixed depth had been recorded at 12.9, 16.6 and 18.3 tokens/s
across boots, a 42 percent spread with no explanation, while nothing else in
the campaign moved by more than about 1 percent. `variance/` holds ten
consecutive runs in one boot with the shader clock sampled during each, and
`history/` tests whether what ran beforehand is the variable.

## 2. The SDMA completion path

Earlier work narrowed the SDMA failure to one step: ROCclr asks the runtime for
an async copy above 16384 bytes, the runtime routes it to SDMA, and the
completion signal never fires while no trap interrupt is dispatched. What was
not known is whether the engine consumed the packet and failed to signal, or
never advanced at all. `sdma-mqd/` samples the KFD queue descriptors during a
hanging copy to separate those.

## 3. Endurance on the current stack

The two soaks in this repository predate the corrected patch scripts, the
native PyTorch build and the llama.cpp patch set as it now stands. `soak/` runs
the configuration a reader would build today, and adds a PyTorch training loop
every third round so it covers the paths the earlier soaks never touched.

## 4. The zeroed fp16 GEMM

Still open, and not reproducible in a standalone program or in PyTorch on the
same rocBLAS. `fp16-graphs/` tests an angle never tried: every observation of
this defect comes from a run with HIP graph capture enabled, so the arms compare
graphs on against graphs off, with the f32 compute type as a control and a
quantized-value model that should never take the path at all.

Results and what they change are in the sections of the top-level README that
each item belongs to; this directory holds the raw runs.


## Outcomes

**1. The 8B decode variance: characterised and explained.** The spread is
intrinsic to decode at depth, coefficient of variation 4.3 percent over ten runs
in one boot, with clock, temperature and free memory all steady. It is not the
model or the board: prefill on the same model varies 1.1 percent. The
explanation is bandwidth headroom, and `controls/` demonstrates it rather than
inferring it, by running the identical measurement on a model at 29 percent of
the memory ceiling instead of 80: coefficient of variation 0.6 percent against
4.3, a factor of 6.7 in the predicted direction.

A correction came out of this. The 12.9 figure previously quoted for this model
exists in no shipped log; it was taken interactively and never captured. The
range from captured evidence is 16.21 to 18.71, and `history/` shows that prior
heavy work, the obvious explanation for a low outlier, does not reproduce it.

**2. The SDMA completion path: narrowed to the submission end.** The queue
descriptor is byte-identical throughout a hanging copy, against a control
showing descriptors change substantially during real work. So the queue never
advances, and the failure is between the runtime handing over the copy and the
queue being rung rather than the engine completing work and failing to signal.

**3. Endurance on the current stack: nothing moved.** 42 rounds over eight
hours four minutes on the configuration this document now recommends. One
distinct perplexity value across every round, 8.9442; prefill spread 0.35
percent; 42 of 42 churn sweeps clean; zero GPU faults. The fourteen PyTorch
training runs, the path no earlier soak covered, returned an identical final
loss every time.

**4. The zeroed fp16 GEMM: two hypotheses removed, and the pattern sharpened.**
HIP graph capture is not involved, and neither is flash attention; the defect
survives both structural changes. Counting the earlier per-tensor dumps shows
it is deterministic rather than intermittent within a run: across five forward
passes of 36 value projections, the first pass is clean and every subsequent
pass has its first projection zeroed, with no other projection affected. The
control that the defect never touches models with quantized value weights now
uses the right model, deepseek-r1-14B, which returns one bit-identical value
across six runs regardless of compute type.
