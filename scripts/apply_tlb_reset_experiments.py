#!/usr/bin/env python3
# One experiment module for the September 2026 round, every arm behind a runtime
# parameter so the arms can be compared within one boot. Applied on top of the
# production tree (runlist flush, SVM map-side flush, flush_pasid_kiq param,
# 40-CU unlock). All defaults reproduce production behaviour.
#
# A. Lighter stale-TLB fixes than the runlist rebuild.
#    The August VMID-flush candidates (logs/vmid-flush-2026-08-20) called
#    gmc_v10_0_flush_gpu_tlb(), which takes the KIQ branch whenever the KIQ ring
#    is ready, i.e. always at runtime. So "direct invalidation cannot substitute"
#    was measured for the KIQ route only. These arms bypass it.
#
#    amdgpu.bc250_tlb_alt   0  production: runlist rebuild at the sites selected by
#                              bc250_flush_by_runlist
#                           1  MMIO, bypassing the KIQ: invalidate every KFD VMID
#                              (first_kfd_vmid..15) on engine 17, poll the ACK
#                           2  MMIO, only VMIDs whose GCVM_CONTEXTn page-table base
#                              equals this process's page directory
#                           3  SDMA VM_INVALIDATION packet, every KFD VMID
#                           4  SDMA VM_INVALIDATION packet, matching VMIDs only
#                           Replaces the rebuild at the same sites; needs
#                           bc250_flush_by_runlist to select them.
#    amdgpu.bc250_tlb_dump  N  log the page-table base of VMIDs 1..15 and the
#                              process page directory on the next N flush calls
#    amdgpu.bc250_tlb_ack_us   MMIO ACK poll budget per VMID, default 2000
#    Statistics are logged as BC250TLBALT lines.
#
# B. amdgpu.bc250_honest_reset=1: a MODE1 reset with no PSP mode1_reset, or a
#    MODE2 reset with no SMU mode2_reset, returns -EOPNOTSUPP instead of success.
#
# C. amdgpu_device_should_recover_gpu() checks the per-ASIC default-disable list
#    before the RAS early return, so gpu_recovery=-1 disables recovery on Cyan
#    Skillfish as the list intends. Unconditional (only visible with the default).
#    amdgpu.bc250_should_recover_probe: writing anything logs the verdict.
#
# Usage: apply_tlb_reset_experiments.py <tree>/drivers/gpu/drm/amd
import sys, pathlib, re

amd = pathlib.Path(sys.argv[1])
MARK = "bc250_tlb_alt"

def edit(rel, fn):
    p = amd / rel
    s = p.read_text()
    out = fn(s)
    assert out is not None and out != s, f"no change applied to {rel}"
    p.write_text(out)
    print("patched", rel)

if MARK in (amd / "amdkfd/kfd_device_queue_manager.c").read_text():
    print("already patched"); sys.exit(0)

# ---------------------------------------------------------------- A: amdgpu side
AMDKFD_C = r'''
/* ---- BC-250 experiment, see scripts/apply_tlb_reset_experiments.py ---- */
#include "navi10_sdma_pkt_open.h"

static struct amdgpu_device *bc250_exp_adev;

int amdgpu_amdkfd_bc250_tlb_inv(struct amdgpu_device *adev, u64 pd, int mode,
				int dump, int ack_us, u32 *out_matched,
				u32 *out_acked)
{
	struct amdgpu_vmhub *hub = &adev->vmhub[AMDGPU_GFXHUB(0)];
	const unsigned int eng = 17;
	u32 req = hub->vm_inv_eng0_req + hub->eng_distance * eng;
	u32 ack = hub->vm_inv_eng0_ack + hub->eng_distance * eng;
	u32 first = adev->vm_manager.first_kfd_vmid;
	u32 matched = 0, acked = 0, targets = 0;
	int vmid, i, r = 0;
	u64 ptb;

	bc250_exp_adev = adev;
	for (vmid = 1; vmid < 16; vmid++) {
		ptb = ((u64)RREG32(hub->ctx0_ptb_addr_hi32 + hub->ctx_addr_distance * vmid) << 32) |
		      RREG32(hub->ctx0_ptb_addr_lo32 + hub->ctx_addr_distance * vmid);
		if (pd && (ptb & ~0xfffULL) == (pd & ~0xfffULL))
			matched |= 1U << vmid;
		if (dump)
			dev_info(adev->dev, "BC250TLB vmid=%d ptb=0x%llx pd=0x%llx first_kfd=%u%s\n",
				 vmid, ptb, pd, first, (matched & (1U << vmid)) ? " MATCH" : "");
	}

	if (mode == 1 || mode == 3)
		targets = (0xffffU << first) & 0xfffeU;
	else if (mode == 2 || mode == 4)
		targets = matched;

	if ((mode == 1 || mode == 2) && targets) {
		spin_lock(&adev->gmc.invalidate_lock);
		for (vmid = 1; vmid < 16; vmid++) {
			u32 inv;

			if (!(targets & (1U << vmid)))
				continue;
			inv = hub->vmhub_funcs->get_invalidate_req(vmid, TLB_FLUSH_HEAVYWEIGHT);
			WREG32_RLC_NO_KIQ(req, inv, GC_HWIP);
			/* dummy read, as gmc_v10_0_flush_gpu_tlb does below GC 10.3 */
			RREG32_RLC_NO_KIQ(req, GC_HWIP);
			for (i = 0; i < ack_us; i++) {
				if (RREG32_RLC_NO_KIQ(ack, GC_HWIP) & (1U << vmid))
					break;
				udelay(1);
			}
			if (i < ack_us)
				acked |= 1U << vmid;
		}
		spin_unlock(&adev->gmc.invalidate_lock);
	} else if ((mode == 3 || mode == 4) && targets) {
		struct amdgpu_ring *ring = adev->mman.buffer_funcs_ring;
		struct dma_fence *fence;
		struct amdgpu_job *job;
		long t;

		if (!ring || !ring->sched.ready || !adev->mman.buffer_funcs_enabled ||
		    !adev->ib_pool_ready)
			return -ENODEV;
		mutex_lock(&adev->mman.default_entity.lock);
		r = amdgpu_job_alloc_with_ib(adev, &adev->mman.default_entity.base,
					     AMDGPU_FENCE_OWNER_UNDEFINED, 80 * 4,
					     AMDGPU_IB_POOL_IMMEDIATE, &job,
					     AMDGPU_KERNEL_JOB_ID_FLUSH_GPU_TLB);
		if (r) {
			mutex_unlock(&adev->mman.default_entity.lock);
			return r;
		}
		for (vmid = 1; vmid < 16; vmid++) {
			struct amdgpu_ib *ib = &job->ibs[0];

			if (!(targets & (1U << vmid)))
				continue;
			/* packet layout as in the sdma_v5_0_emit_tlb_inv proposed on
			 * amd-gfx, 2026-09-11 */
			ib->ptr[ib->length_dw++] =
				SDMA_PKT_VM_INVALIDATION_HEADER_OP(SDMA_OP_POLL_REGMEM) |
				SDMA_PKT_VM_INVALIDATION_HEADER_SUB_OP(SDMA_SUBOP_VM_INVALIDATION) |
				SDMA_PKT_VM_INVALIDATION_HEADER_GFX_ENG_ID(eng) |
				SDMA_PKT_VM_INVALIDATION_HEADER_MM_ENG_ID(0x1f);
			ib->ptr[ib->length_dw++] =
				hub->vmhub_funcs->get_invalidate_req(vmid, TLB_FLUSH_HEAVYWEIGHT);
			ib->ptr[ib->length_dw++] = 0xFFFFFFFF;
			ib->ptr[ib->length_dw++] =
				SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_INVALIDATEACK(1U << vmid) |
				SDMA_PKT_VM_INVALIDATION_ADDRESSRANGEHI_ADDRESSRANGEHI(0x1F);
		}
		amdgpu_ring_pad_ib(ring, &job->ibs[0]);
		fence = amdgpu_job_submit(job);
		mutex_unlock(&adev->mman.default_entity.lock);
		t = dma_fence_wait_timeout(fence, false, msecs_to_jiffies(3000));
		dma_fence_put(fence);
		if (t <= 0)
			r = -ETIME;
		else
			acked = targets;
	}

	*out_matched = matched;
	*out_acked = acked;
	return r;
}

static int bc250_should_recover_probe_set(const char *val, const struct kernel_param *kp)
{
	if (!bc250_exp_adev) {
		pr_warn("BC250RECOVER: no device recorded yet, run a GPU flush first\n");
		return -ENODEV;
	}
	pr_warn("BC250RECOVER: gpu_recovery=%d should_recover=%d\n", amdgpu_gpu_recovery,
		amdgpu_device_should_recover_gpu(bc250_exp_adev) ? 1 : 0);
	return 0;
}

static const struct kernel_param_ops bc250_should_recover_probe_ops = {
	.set = bc250_should_recover_probe_set,
};
module_param_cb(bc250_should_recover_probe, &bc250_should_recover_probe_ops, NULL, 0644);
'''
edit("amdgpu/amdgpu_amdkfd.c", lambda s: s + AMDKFD_C)
edit("amdgpu/amdgpu_amdkfd.h", lambda s: s.replace(
    "#endif /* AMDGPU_AMDKFD_H_INCLUDED */",
    "/* BC-250 experiment */\nint amdgpu_amdkfd_bc250_tlb_inv(struct amdgpu_device *adev, u64 pd, int mode,\n"
    "\t\t\t\tint dump, int ack_us, u32 *out_matched, u32 *out_acked);\n\n"
    "#endif /* AMDGPU_AMDKFD_H_INCLUDED */"))

# ---------------------------------------------------------------- A: KFD side
DQM_C = r'''
/* ---- BC-250 experiment, see scripts/apply_tlb_reset_experiments.py ---- */
static int bc250_tlb_alt;
module_param(bc250_tlb_alt, int, 0644);
MODULE_PARM_DESC(bc250_tlb_alt, "BC-250: 0 runlist, 1 MMIO kfd vmids, 2 MMIO matched, 3 SDMA kfd vmids, 4 SDMA matched");
static int bc250_tlb_dump;
module_param(bc250_tlb_dump, int, 0644);
static int bc250_tlb_ack_us = 2000;
module_param(bc250_tlb_ack_us, int, 0644);

static atomic_t bc250_alt_calls = ATOMIC_INIT(0);
static atomic_t bc250_alt_nomatch = ATOMIC_INIT(0);
static atomic_t bc250_alt_ackmiss = ATOMIC_INIT(0);
static atomic_t bc250_alt_err = ATOMIC_INIT(0);

void kfd_bc250_flush_alt(struct kfd_process_device *pdd, int site)
{
	struct kfd_node *dev = pdd ? pdd->dev : NULL;
	u32 matched = 0, acked = 0;
	int dump = 0, r, n;
	u64 pd;

	if (!dev || !dev->adev || !dev->adev->pdev ||
	    dev->adev->pdev->device != BC250_PCI_DEVICE_ID ||
	    (!bc250_tlb_alt && bc250_tlb_dump <= 0)) {
		kfd_bc250_flush_by_runlist(dev, site);
		return;
	}
	pd = amdgpu_amdkfd_gpuvm_get_process_page_dir(pdd->drm_priv);
	if (bc250_tlb_dump > 0) {
		dump = 1;
		bc250_tlb_dump--;
	}
	if (!bc250_tlb_alt || !(bc250_flush_by_runlist & site)) {
		if (dump)
			amdgpu_amdkfd_bc250_tlb_inv(dev->adev, pd, 0, 1, 0, &matched, &acked);
		kfd_bc250_flush_by_runlist(dev, site);
		return;
	}
	r = amdgpu_amdkfd_bc250_tlb_inv(dev->adev, pd, bc250_tlb_alt, dump,
					bc250_tlb_ack_us, &matched, &acked);
	n = atomic_inc_return(&bc250_alt_calls);
	if (!matched)
		atomic_inc(&bc250_alt_nomatch);
	if (r)
		atomic_inc(&bc250_alt_err);
	else if (bc250_tlb_alt <= 2 && acked != ((bc250_tlb_alt == 1) ?
			((0xffffU << dev->adev->vm_manager.first_kfd_vmid) & 0xfffeU) : matched))
		atomic_inc(&bc250_alt_ackmiss);
	if (dump || n == 1 || n == 10 || n == 100 || (n % 1000) == 0 || r)
		dev_info(dev->adev->dev,
			 "BC250TLBALT mode=%d site=%d calls=%d nomatch=%d ackmiss=%d err=%d last: matched=0x%04x acked=0x%04x r=%d\n",
			 bc250_tlb_alt, site, n, atomic_read(&bc250_alt_nomatch),
			 atomic_read(&bc250_alt_ackmiss), atomic_read(&bc250_alt_err),
			 matched, acked, r);
}
'''
edit("amdkfd/kfd_device_queue_manager.c", lambda s: s + DQM_C)
edit("amdkfd/kfd_device_queue_manager.h", lambda s: s.replace(
    "int kfd_bc250_flush_by_runlist(struct kfd_node *dev, int site);",
    "struct kfd_process_device;\nvoid kfd_bc250_flush_alt(struct kfd_process_device *pdd, int site); /* bc250_tlb_alt */\n"
    "int kfd_bc250_flush_by_runlist(struct kfd_node *dev, int site);", 1))

def sites(s):
    s = re.sub(r"kfd_bc250_flush_by_runlist\(peer_pdd->dev, (\d)\)", r"kfd_bc250_flush_alt(peer_pdd, \1)", s)
    s = re.sub(r"kfd_bc250_flush_by_runlist\(pdd->dev, (\d)\)", r"kfd_bc250_flush_alt(pdd, \1)", s)
    return s
edit("amdkfd/kfd_chardev.c", sites)
edit("amdkfd/kfd_svm.c", sites)

# ---------------------------------------------------------------- B and C
def device_c(s):
    old = '''	} else {
		dev_info(adev->dev, "GPU psp mode1 reset\\n");
		ret = psp_gpu_reset(adev);
	}'''
    new = '''	} else {
		dev_info(adev->dev, "GPU psp mode1 reset\\n");
		if (bc250_honest_reset && (!adev->psp.funcs || !adev->psp.funcs->mode1_reset)) {
			dev_err(adev->dev, "BC250: PSP has no mode1_reset for this ASIC\\n");
			ret = -EOPNOTSUPP;
		} else {
			ret = psp_gpu_reset(adev);
		}
	}'''
    assert old in s, "mode1 anchor"
    s = s.replace(old, new, 1)
    s = s.replace("int amdgpu_device_mode1_reset(struct amdgpu_device *adev)\n{",
        "/* BC-250 experiment, see scripts/apply_tlb_reset_experiments.py */\n"
        "int bc250_honest_reset;\nmodule_param(bc250_honest_reset, int, 0644);\n\n"
        "int amdgpu_device_mode1_reset(struct amdgpu_device *adev)\n{", 1)
    # C: move the default-disable list ahead of the RAS early return
    m = re.search(r"(\tif \(amdgpu_gpu_recovery == 0\)\n\t\tgoto disabled;\n)"
                  r"(\n\t/\* Skip soft reset check in fatal error mode \*/\n\tif \(!amdgpu_ras_is_poison_mode_supported\(adev\)\)\n\t\treturn true;\n)"
                  r"(\n\tif \(amdgpu_sriov_vf\(adev\)\)\n\t\treturn true;\n)"
                  r"(\n\tif \(amdgpu_gpu_recovery == -1\) \{.*?\n\t\}\n)", s, re.S)
    assert m, "should_recover anchor"
    s = s.replace(m.group(0), m.group(1) + m.group(3) + m.group(4) + m.group(2), 1)
    return s
edit("amdgpu/amdgpu_device.c", device_c)

def smu_c(s):
    old = "\tif (smu->ppt_funcs->mode2_reset)\n\t\tret = smu->ppt_funcs->mode2_reset(smu);\n"
    assert old in s, "mode2 anchor"
    new = ("\tif (smu->ppt_funcs->mode2_reset)\n\t\tret = smu->ppt_funcs->mode2_reset(smu);\n"
           "\telse if (bc250_honest_reset) /* BC-250 experiment */\n\t\tret = -EOPNOTSUPP;\n")
    s = s.replace(old, new, 1)
    return s.replace("static int smu_mode2_reset(void *handle)",
                     "extern int bc250_honest_reset;\nstatic int smu_mode2_reset(void *handle)", 1)
edit("pm/swsmu/amdgpu_smu.c", smu_c)
print("all applied")
