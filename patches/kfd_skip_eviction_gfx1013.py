#!/usr/bin/env python3
# BC-250 experiment: skip KFD queue eviction on gfx1013 (device 0x13FE) in the
# nocpsch path (sched_policy=2), replicating the upstream gfx1103 APU workaround.
# The munmap-triggered eviction is what times out the MEC preemption and wedges;
# skipping it should avoid the wedge (at the cost of overcommit correctness).
import re, sys
f = "/usr/src/linux-6.18.9/drivers/gpu/drm/amd/amdkfd/kfd_device_queue_manager.c"
s = open(f).read()
guard = ("\tif (((struct amdgpu_device *)dqm->dev->adev)->pdev->device == 0x13FE)"
         " /* BC-250: skip queue eviction (avoids MEC preempt wedge) */\n\t\treturn 0;\n")
n = 0
for fn in ["evict_process_queues_nocpsch", "restore_process_queues_nocpsch"]:
    head = s[s.find("int " + fn): s.find("int " + fn) + 400]
    if "0x13FE" in head:
        print(fn, "already patched"); continue
    pat = re.compile(r"static int " + fn + r"\(.*?\)\n\{\n", re.DOTALL)
    s2 = pat.sub(lambda m: m.group(0) + guard, s, count=1)
    if s2 != s:
        n += 1; s = s2; print(fn, "patched")
    else:
        print(fn, "NO MATCH")
open(f, "w").write(s)
print("total patched:", n)
