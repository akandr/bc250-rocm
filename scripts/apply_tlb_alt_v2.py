#!/usr/bin/env python3
# Second revision of the stale-TLB experiment, on top of apply_tlb_reset_experiments.py.
#
# The first round (logs/tlb-alt-2026-09-15/tbo/) found that SDMA invalidation of
# the VMID the firmware assigned completes without error and still does not stop
# the churn fault, while MMIO invalidation never acknowledges. Two additions:
#
#   bc250_tlb_alt=5   rebuild the runlist for this process only
#                     (execute_queues_cpsch with KFD_UNMAP_QUEUES_FILTER_BY_PASID),
#                     the lighter form of the reassignment that does work
#   SDMA modes 3/4    report the GCVM_INVALIDATE_ENG17_ACK register read after the
#                     job completes, instead of assuming completion means ACK,
#                     since the SDMA poll packet can give up without it
#
# Usage: apply_tlb_alt_v2.py <tree>/drivers/gpu/drm/amd   (idempotent)
import sys, pathlib
amd = pathlib.Path(sys.argv[1])

p = amd / "amdgpu/amdgpu_amdkfd.c"; s = p.read_text()
if "BC250 v2 ack readback" not in s:
    old = "\t\tif (t <= 0)\n\t\t\tr = -ETIME;\n\t\telse\n\t\t\tacked = targets;\n"
    assert old in s, "sdma completion anchor"
    s = s.replace(old, "\t\tif (t <= 0)\n\t\t\tr = -ETIME;\n"
                  "\t\t/* BC250 v2 ack readback: completion of the job is not proof of ACK */\n"
                  "\t\tacked = RREG32_RLC_NO_KIQ(ack, GC_HWIP) & targets;\n", 1)
    p.write_text(s); print("patched", p.name)

p = amd / "amdkfd/kfd_device_queue_manager.c"; s = p.read_text()
if "BC250 v2 pasid rebuild" not in s:
    old = "\tr = amdgpu_amdkfd_bc250_tlb_inv(dev->adev, pd, bc250_tlb_alt, dump,\n"
    assert old in s, "alt call anchor"
    new = ("\tif (bc250_tlb_alt == 5) { /* BC250 v2 pasid rebuild */\n"
           "\t\tstruct device_queue_manager *dqm = dev->dqm;\n\n"
           "\t\tif (!dqm || dqm->sched_policy == KFD_SCHED_POLICY_NO_HWS)\n\t\t\treturn;\n"
           "\t\tdqm_lock(dqm);\n"
           "\t\tr = execute_queues_cpsch(dqm, KFD_UNMAP_QUEUES_FILTER_BY_PASID, pdd->pasid,\n"
           "\t\t\t\t\t USE_DEFAULT_GRACE_PERIOD);\n"
           "\t\tdqm_unlock(dqm);\n"
           "\t\tn = atomic_inc_return(&bc250_alt_calls);\n"
           "\t\tif (r)\n\t\t\tatomic_inc(&bc250_alt_err);\n"
           "\t\tif (n == 1 || n == 10 || n == 100 || (n % 1000) == 0 || r)\n"
           "\t\t\tdev_info(dev->adev->dev, \"BC250TLBALT mode=5 site=%d calls=%d err=%d pasid=%u r=%d\\n\",\n"
           "\t\t\t\t site, n, atomic_read(&bc250_alt_err), pdd->pasid, r);\n"
           "\t\treturn;\n\t}\n" + old)
    s = s.replace(old, new, 1)
    p.write_text(s); print("patched", p.name)
print("v2 applied")
