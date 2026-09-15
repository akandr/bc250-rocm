#!/usr/bin/env python3
# Third revision of the stale-TLB experiment: what exactly does the runlist rebuild do that
# invalidation does not? On top of apply_tlb_reset_experiments.py and apply_tlb_alt_v2.py.
#
# Established so far (logs/tlb-alt-2026-09-15/): the rebuild clears the stale translation; an
# MMIO invalidation of the VMID the firmware assigned is never acknowledged; an SDMA
# invalidation completes with the ACK bit clear. MMIO invalidation does acknowledge at boot,
# before the firmware runs (no "VM flush ACK" timeout is ever logged then). Two readings follow,
# and each is also a candidate one-register fix:
#
#   - the rebuild's MAP_PROCESS makes the firmware write the VMID's page-table base, and a write
#     to GCVM_CONTEXTn_PAGE_TABLE_BASE_ADDR may flush that context by itself;
#   - once the firmware owns the hub, the invalidation engine may need RLC safe mode or its
#     semaphore, or a different engine, before a request is taken.
#
#   amdgpu.bc250_tlb_alt=6   rewrite the matched VMID's page-table base with its current value
#   amdgpu.bc250_tlb_alt=7   write the matched VMID's page-table base to 0, then restore it
#   amdgpu.bc250_tlb_alt=8   mode 6 followed by the MMIO invalidation of mode 9
#   amdgpu.bc250_tlb_alt=9   MMIO invalidation of the matched VMID, shaped by the two below
#   amdgpu.bc250_tlb_flags   1 = RLC safe mode around the access, 2 = take the engine semaphore,
#                            4 = log the request and ACK registers read back (BC250INV lines)
#   amdgpu.bc250_tlb_eng     invalidation engine for mode 9 (default 17; the kernel's rings
#                            use 0 to 13 on this board)
#
# Usage: apply_tlb_alt_v3.py <tree>/drivers/gpu/drm/amd   (idempotent)
import sys, pathlib
p = pathlib.Path(sys.argv[1]) / "amdgpu/amdgpu_amdkfd.c"
s = p.read_text()
if "bc250_tlb_flags" in s:
    print("already patched"); sys.exit(0)

helpers = r'''
/* ---- BC-250 experiment v3, see scripts/apply_tlb_alt_v3.py ---- */
static int bc250_tlb_eng = 17;
module_param(bc250_tlb_eng, int, 0644);
static int bc250_tlb_flags;
module_param(bc250_tlb_flags, int, 0644);

static void bc250_mmio_inv(struct amdgpu_device *adev, struct amdgpu_vmhub *hub, u32 vmid,
			   int ack_us, u32 *acked, int dump)
{
	u32 eng = bc250_tlb_eng;
	u32 sem = hub->vm_inv_eng0_sem + hub->eng_distance * eng;
	u32 req = hub->vm_inv_eng0_req + hub->eng_distance * eng;
	u32 ack = hub->vm_inv_eng0_ack + hub->eng_distance * eng;
	u32 inv = hub->vmhub_funcs->get_invalidate_req(vmid, TLB_FLUSH_HEAVYWEIGHT);
	u32 rb_req, ack0, ack1 = 0;
	int i, sem_ok = -1;

	spin_lock(&adev->gmc.invalidate_lock);
	if (bc250_tlb_flags & 2) {
		for (i = 0; i < ack_us; i++) {
			if (RREG32_RLC_NO_KIQ(sem, GC_HWIP) & 0x1)
				break;
			udelay(1);
		}
		sem_ok = i < ack_us;
	}
	ack0 = RREG32_RLC_NO_KIQ(ack, GC_HWIP);
	WREG32_RLC_NO_KIQ(req, inv, GC_HWIP);
	rb_req = RREG32_RLC_NO_KIQ(req, GC_HWIP);
	for (i = 0; i < ack_us; i++) {
		ack1 = RREG32_RLC_NO_KIQ(ack, GC_HWIP);
		if (ack1 & (1U << vmid))
			break;
		udelay(1);
	}
	if (i < ack_us)
		*acked |= 1U << vmid;
	if (bc250_tlb_flags & 2)
		WREG32_RLC_NO_KIQ(sem, 0, GC_HWIP);
	spin_unlock(&adev->gmc.invalidate_lock);

	if (dump || (bc250_tlb_flags & 4))
		dev_info_ratelimited(adev->dev,
			"BC250INV eng=%u vmid=%u wrote=0x%08x req_readback=0x%08x ack_before=0x%08x ack_after=0x%08x sem_ok=%d polls=%d\n",
			eng, vmid, inv, rb_req, ack0, ack1, sem_ok, i);
}

static void bc250_ptb_touch(struct amdgpu_device *adev, struct amdgpu_vmhub *hub, u32 vmid,
			    bool through_zero, int dump)
{
	u32 lo_reg = hub->ctx0_ptb_addr_lo32 + hub->ctx_addr_distance * vmid;
	u32 hi_reg = hub->ctx0_ptb_addr_hi32 + hub->ctx_addr_distance * vmid;
	u32 lo = RREG32(lo_reg), hi = RREG32(hi_reg);

	if (through_zero) {
		WREG32(lo_reg, 0);
		WREG32(hi_reg, 0);
	}
	WREG32(lo_reg, lo);
	WREG32(hi_reg, hi);
	if (dump)
		dev_info(adev->dev, "BC250PTB vmid=%u rewrote 0x%08x%08x%s, readback 0x%08x%08x\n",
			 vmid, hi, lo, through_zero ? " via zero" : "", RREG32(hi_reg), RREG32(lo_reg));
}
'''
anchor_fn = "int amdgpu_amdkfd_bc250_tlb_inv(struct amdgpu_device *adev, u64 pd, int mode,"
assert anchor_fn in s, "function anchor"
s = s.replace(anchor_fn, helpers + "\n" + anchor_fn, 1)

old_t = "\telse if (mode == 2 || mode == 4)\n\t\ttargets = matched;\n"
assert old_t in s, "targets anchor"
new_t = old_t + r'''
	if (mode >= 6 && mode <= 9) {
		bool safe = bc250_tlb_flags & 1;

		if (!matched)
			goto out_v3;
		if (safe)
			amdgpu_gfx_rlc_enter_safe_mode(adev, 0);
		for (vmid = 1; vmid < 16; vmid++) {
			if (!(matched & (1U << vmid)))
				continue;
			if (mode == 6 || mode == 8)
				bc250_ptb_touch(adev, hub, vmid, false, dump);
			if (mode == 7)
				bc250_ptb_touch(adev, hub, vmid, true, dump);
			if (mode == 8 || mode == 9)
				bc250_mmio_inv(adev, hub, vmid, ack_us, &acked, dump);
		}
		if (safe)
			amdgpu_gfx_rlc_exit_safe_mode(adev, 0);
out_v3:
		*out_matched = matched;
		*out_acked = acked;
		return 0;
	}
'''
s = s.replace(old_t, new_t, 1)
p.write_text(s)

# the KFD-side ack-miss accounting only knows modes 1 and 2; extend it so modes 8 and 9 report
q = pathlib.Path(sys.argv[1]) / "amdkfd/kfd_device_queue_manager.c"
t = q.read_text()
old = "\telse if (bc250_tlb_alt <= 2 && acked != ((bc250_tlb_alt == 1) ?"
if old in t:
    t = t.replace(old, "\telse if ((bc250_tlb_alt <= 2 || bc250_tlb_alt >= 8) && acked != ((bc250_tlb_alt == 1) ?", 1)
    q.write_text(t)
print("v3 applied")
