**Independent confirmation, on this board, of a mechanism GabriWar identified first.**

Credit where it belongs, and this repository already recorded it: the
bc250-rocm-working project found by driver instrumentation that
`gmc_v10_0_flush_gpu_tlb_pasid()` looks for the owning VMID in a register that
gfx10 under hardware scheduling never writes, so the flush matches nothing, with
20 of 20 flushes hitting zero VMIDs in their measurements. What follows is the
same thing measured here, which is worth having as a second board and a second
instrument but is not a new finding. An earlier version of this note presented it
as one, which was wrong; the document it sits in already carried the
attribution.

**The mechanism, confirmed at source level.**

This answers a question GabriWar asked: whether there is a better way to fix the
translation than rebuilding the whole runlist. The diagnosis is now certain; the
obvious fix built on it is not viable, and that is reported too.

**The mechanism.** `kfd_flush_tlb()` calls `amdgpu_vm_flush_compute_tlb()`, which
is already scoped to one VM, and that calls `amdgpu_gmc_flush_gpu_tlb_pasid()`.
On gfx10 this is `gmc_v10_0_flush_gpu_tlb_pasid()`, which walks VMIDs 1 to 15
asking `gmc_v10_0_get_atc_vmid_pasid_mapping_info()` which PASID each holds, and
skips any that do not match:

    valid = gmc_v10_0_get_atc_vmid_pasid_mapping_info(adev, vmid, &queried);
    if (!valid || queried != pasid)
            continue;
    gmc_v10_0_flush_gpu_tlb(adev, vmid, AMDGPU_GFXHUB(0), flush_type);

That helper reads `ATC_VMID*_PASID_MAPPING` from the ATHUB and returns its VALID
bit.

**The measurement.** A module built with a counter around that loop reports, on
every call without exception:

    BC250FLUSH call=1 pasid=15 valid_vmids=0 matches=0 total_matched=0
    BC250FLUSH call=2 pasid=15 valid_vmids=0 matches=0 total_matched=0
    ...

`valid_vmids=0 matches=0` is the only pair ever observed. What backs "ever" is
the instrument's own distinct-value summary rather than an exhaustive listing:
`instrumentation.txt` ships six calls, labelled a sample under load, followed by
the line reporting that one pair was the only one seen. How many calls that
summary covers was not recorded, noted 26 August. It is a strong result either
way, since a single counterexample would have shown up as a second pair, but the
denominator is not in the file. Not one VMID reports a
valid ATC mapping, so the loop body never executes and the function returns
having flushed nothing, with no error and nothing logged. The PASID flush on this
ASIC is a silent no-op, which is exactly the symptom this repository has called
"the PASID invalidation covers nothing" since July, now with a mechanism.

**The obvious fix does not work.** If the lookup never matches, the natural
repair is to invalidate the userspace VMIDs directly instead. Implemented behind
a runtime parameter and tested: it is catastrophically slow, the allocation-reuse
reproducer failed 3 of 3, and leaving it enabled rebooted the board twice. Two
likely reasons, neither verified: the flush is called often enough that fifteen
register-driven invalidations per call is a heavy tax, and VMIDs 1 to 15 include
those in use by graphics contexts, which this blindly invalidates.

**What the next attempt should do.** The KFD tracks which VMIDs it owns, in
`struct kfd_vmid_info` as `first_vmid_kfd` and `last_vmid_kfd`. Restricting the
fallback to that range would avoid touching graphics VMIDs and cut the count.
That is untested.

**A caveat on the harness.** The reproducer used here, `seq_probe`, ran clean at
`bc250_flush_by_runlist=0` as well, so it was not discriminating on the current
stack and cannot show that a candidate fix works. The load that does discriminate
is a model load, which aborts with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`
at `runlist=0` (see `../flush-cost-2026-08-18/`). Any future test of this should
use that instead.

The board was returned to a clean module afterwards: instrumentation and fallback
removed, `bc250_flush_pasid_kiq` and the runlist patches intact, gate 8.9442,
zero faults.

That sentence used to end "pp512 805.53, tg64 113.50" as well. Those are the
standing reference figures for this model, not a measurement kept from that
evening, and no restoration run survives here. The only throughput captured on
this board that night is in `../kernel-718-2026-08-19/`, taken a few hours later
on a clean module, and it reads 808.17 and 113.53. Whether the restoration was
separately measured and matched, or the reference was simply restated, the record
does not say, so the claim is now limited to what the record supports: the gate
returned its established value and no faults were seen.
