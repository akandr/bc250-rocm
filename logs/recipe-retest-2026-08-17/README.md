# Retesting two recipe restrictions, 2026-08-17

Both were written before the allocation-reuse flush existed, and both are now
unnecessary.

**Memory mapping.** The recipe required `--no-mmap`. With mapping enabled the
1.5B returns perplexity 8.9442 on three of three runs, bit-identical to the
no-mmap reference, and the 14B loads three of three.

The 14B half of that was captured and the 1.5B half was not, noted 26 August, and
the missing half has since been measured: three runs with mapping enabled return
8.9442 plus or minus 0.17287, bit-identical to each other and to a no-mmap
control run in the same session
([`mmap-gate-2026-08-26/`](mmap-gate-2026-08-26/)). The claim was right. What
follows is what was and was not on record before that run. The
three `mmap_q14_*.log` files hold successful loads, rc=0 each. The three
`mmap_q15_*.log` files hold the argument error described below, and the only
`8.9442` anywhere in this directory's captures is the reference quoted in a
section header of `log`, which is not a measurement. The corrected 1.5B runs were
not kept, and "the corrected runs are in this README" below should be read as
saying exactly that rather than as a pointer.
Nor is the claim corroborated elsewhere in the way it might appear to be: every
shipped perplexity harness in this repository passes `--no-mmap`, so no captured
gate anywhere runs with mapping enabled. The recipe change stands on the 14B
loads and on the throughput work that has used mapping ever since without
incident; the specific bit-identical perplexity claim stands on a run that was
not kept. Note the flag situation:
`--mmap` and `--no-mmap` are deprecated in current llama.cpp in favour of
`--load-mode`, and `--mmap 1` is rejected because the deprecated flag is a
boolean. The three failing runs in `log` with rc=1 are that argument error, not
a load failure; the corrected runs are in this README.

**One benchmark per invocation.** The recipe warned that multi-size sweeps
reallocate between tests and can trip the load-time fault mid-run. A single
invocation sweeping pp128, pp512, pp1024, pp2048, tg32 and tg64 completes all
six rows, with zero faults in dmesg.

The scripts elsewhere in this repo still pass `--no-mmap`, because that is what
they passed when the measurements in this document were taken.
