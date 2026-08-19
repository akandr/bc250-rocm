#!/usr/bin/env python3
"""Apply GabriWar's bc250_flush_by_runlist patch to a kernel tree.

Hand-port of GabriWar/bc250-rocm-working patches/bc250-flush-tlb-by-runlist.patch
(their hunks use pseudo-context, not machine-applicable). Comments translated to
English. Idempotent: skips any file already patched.

Usage: apply_runlist_flush.py [path/to/drivers/gpu/drm/amd/amdkfd]

The directory defaults to the tree this was developed against, which is almost
certainly not yours, so pass it explicitly.
"""
import sys, os, re

DEFAULT_S = os.path.expanduser("~/k715/linux-7.1.5/drivers/gpu/drm/amd/amdkfd")
S = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_S
if not os.path.isdir(S):
    sys.exit("not a directory: %s\n"
             "pass the amdkfd directory of the kernel tree to patch" % S)
if not os.path.exists(os.path.join(S, "kfd_device_queue_manager.c")):
    sys.exit("%s does not look like an amdkfd directory "
             "(no kfd_device_queue_manager.c)" % S)
print("patching tree: %s" % S)
MARK = "kfd_bc250_flush_by_runlist"

DQM_C_ADD = '''
/* BC-250/gfx1013: invalidate the compute TLB by rebuilding the runlist.
 *
 * On this board the compute TLB invalidation never happens: the PASID sweep
 * in gmc_v10_0_flush_gpu_tlb_pasid() depends on mmATC_VMID*_PASID_MAPPING,
 * which on gfx10 under HWS is never written, so it matches zero VMIDs.
 * Forcing the invalidation via KIQ or direct MMIO is never acknowledged and
 * stalls the translation unit. Consequence: hipFree unmaps, hipMalloc reuses
 * the VA with a new physical address, the PTEs are correct in memory, and the
 * GPU keeps translating through the previous mapping.
 *
 * An invalidation that works does exist: the hardware scheduler's runlist
 * rebuild (queue preempt + VMID reassign) must invalidate, or a process would
 * inherit the previous one's translations along with its VMID. This asks for
 * that cycle on unmap so the stale entry never survives to be used.
 *
 * Port of GabriWar/bc250-rocm-working bc250-flush-tlb-by-runlist.patch for
 * validation on kernel 7.1.5 (their measurement base was 7.0.12).
 *
 * Locking: takes dqm_lock. kfd_flush_tlb() must NOT call this --
 * restore_process_queues_nocpsch() and allocate_vmid()/deallocate_vmid()
 * already hold dqm_lock; a deadlock there hangs the machine. The only call
 * site is the unmap ioctl (holds p->mutex, not dqm_lock; same order as
 * pqm_create_queue).
 */
#define BC250_PCI_DEVICE_ID 0x13FE

static int bc250_flush_by_runlist;
module_param(bc250_flush_by_runlist, int, 0644);
MODULE_PARM_DESC(bc250_flush_by_runlist,
	"BC-250: 1=on unmap, invalidate TLB by rebuilding the runlist (default 0)");

int kfd_bc250_flush_by_runlist(struct kfd_node *dev)
{
	struct device_queue_manager *dqm;
	int r;

	if (!bc250_flush_by_runlist || !dev || !dev->adev || !dev->adev->pdev)
		return 0;
	if (dev->adev->pdev->device != BC250_PCI_DEVICE_ID)
		return 0;

	dqm = dev->dqm;
	/* Only meaningful under hardware scheduling: the firmware reassigns
	 * VMIDs when the runlist is rebuilt. Without HWS there is no runlist.
	 */
	if (!dqm || dqm->sched_policy == KFD_SCHED_POLICY_NO_HWS)
		return 0;

	dqm_lock(dqm);
	r = execute_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_DYNAMIC_QUEUES, 0,
				 USE_DEFAULT_GRACE_PERIOD);
	dqm_unlock(dqm);

	if (r)
		dev_err_ratelimited(dev->adev->dev,
			"BC-250: runlist rebuild flush failed (%d)\\n", r);
	return r;
}
'''

DQM_H_ADD = '''
/* BC-250: rebuild the runlist so the firmware invalidates the compute TLB.
 * Full documentation in kfd_device_queue_manager.c above
 * bc250_flush_by_runlist. Takes dqm_lock -- never call with it held.
 */
int kfd_bc250_flush_by_runlist(struct kfd_node *dev);
'''

def patch(path, transform):
    with open(path) as f:
        src = f.read()
    if MARK in src:
        print(f"already patched: {path}")
        return
    out = transform(src)
    if out is None:
        print(f"FAILED anchor: {path}"); sys.exit(1)
    with open(path + ".pre-runlist", "w") as f:
        f.write(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"patched: {path}")

# 1. kfd_device_queue_manager.c -- append param + helper at end of file
patch(os.path.join(S, "kfd_device_queue_manager.c"), lambda s: s + DQM_C_ADD)

# 2. kfd_device_queue_manager.h -- declaration before final #endif
def h_tx(s):
    anchor = "#endif /* KFD_DEVICE_QUEUE_MANAGER_H_ */"
    if anchor not in s: return None
    return s.replace(anchor, DQM_H_ADD + "\n" + anchor)
patch(os.path.join(S, "kfd_device_queue_manager.h"), h_tx)

# 3. kfd_chardev.c -- call site in kfd_ioctl_unmap_memory_from_gpu, after the
#    flush_tlb-conditional kfd_flush_tlb (unique context: dmaunmap comment follows)
def c_tx(s):
    # kfd_flush_tlb takes one argument on 7.x and two on 6.x, so match either
    # rather than only the tree this was written against. Anchoring on the
    # comment that follows keeps the match unique in both cases.
    tail = "\t\t/* Remove dma mapping after tlb flush to avoid IO_PAGE_FAULT */"
    m = re.search(r"\t\tif \(flush_tlb\)\n\t\t\tkfd_flush_tlb\(peer_pdd[^)]*\);\n\n"
                  + re.escape(tail), s)
    if not m: return None
    insert = ("\n"
              "\t\t/* BC-250: rebuild the runlist so the firmware invalidates the\n"
              "\t\t * compute TLB (see kfd_device_queue_manager.c). Called here with\n"
              "\t\t * p->mutex held and dqm_lock free.\n"
              "\t\t */\n"
              "\t\tkfd_bc250_flush_by_runlist(peer_pdd->dev);\n")
    matched = m.group(0)
    repl = matched.replace("\n\n" + tail, insert + "\n" + tail, 1)
    return s.replace(matched, repl, 1)
patch(os.path.join(S, "kfd_chardev.c"), c_tx)

print("all done")
