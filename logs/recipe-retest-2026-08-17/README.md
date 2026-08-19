# Retesting two recipe restrictions, 2026-08-17

Both were written before the allocation-reuse flush existed, and both are now
unnecessary.

**Memory mapping.** The recipe required `--no-mmap`. With mapping enabled the
1.5B returns perplexity 8.9442 on three of three runs, bit-identical to the
no-mmap reference, and the 14B loads three of three. Note the flag situation:
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
