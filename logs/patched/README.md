# Patched-module compute probe, July 2026

The B arm of the Observation 1 comparison. `compute_probe_patched_oberon.log`
is the bare compute probe on a module carrying the corrected PASID flush
(`flush_pasid_uses_kiq = false`), on the same board and workload as the stock
arm in `../stock/`, with the oberon governor running.

Where the stock arm returns wrong results at the larger sizes, this one does
not, which is the A/B behind the claim that the flush change fixes the silent
corruption.

One caveat that the top-level README now states in full: this run came from a
boot where the patched module happened to come up at 40 CU. At the time that
looked like luck, and the explanation offered for it has since been retracted.
The 24-CU boots were a misapplied unlock patch, and 24 CU is not a wedged state
once `amdgpu.sched_policy=2` is off.

Historical boot logs from the patched-module era, collected by hand rather than by a harness.
