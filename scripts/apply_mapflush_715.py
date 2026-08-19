#!/usr/bin/env python3
"""Extend bc250_flush_by_runlist with a MAP-side hook (v2, 2026-08-13).

Evidence (inv28 ftrace): faults fire 44-122 us after a VALID PTE write on a
remapped page. While a page sits unmapped, in-flight work walking neighboring
addresses lets the UTCL2 cache the invalid PTE; the map-side kfd_flush_tlb is
the board's broken PASID sweep (matches zero VMIDs), so the stale-invalid
entry survives and the first GPU touch of the fresh mapping faults. The
unmap-side runlist rebuild cannot help; the MAP side needs the same rebuild.

Param becomes a bitmask, runtime-writable (0644):
  1 = rebuild on unmap (previous behavior)
  2 = rebuild on map
  3 = both (the candidate complete fix)
Idempotent; run on the ALREADY runlist-patched tree.
"""
import sys, os

S = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                       else "~/k715/linux-7.1.5/drivers/gpu/drm/amd/amdkfd")
MARK = "bc250 map-side flush v2"

def patch(path, transform):
    with open(path) as f:
        src = f.read()
    if MARK in src:
        print(f"already patched: {path}")
        return
    out = transform(src)
    if out is None:
        print(f"FAILED anchor: {path}"); sys.exit(1)
    with open(path + ".pre-mapflush", "w") as f:
        f.write(src)
    with open(path, "w") as f:
        f.write(out)
    print(f"patched: {path}")

# 1. helper: take a site mask; check the param bit. (marker comment added)
def dqm_tx(s):
    a1 = 'MODULE_PARM_DESC(bc250_flush_by_runlist,\n\t"BC-250: 1=on unmap, invalidate TLB by rebuilding the runlist (default 0)");'
    r1 = 'MODULE_PARM_DESC(bc250_flush_by_runlist,\n\t"BC-250: bitmask, rebuild runlist to invalidate TLB: 1=on unmap, 2=on map (default 0)"); /* bc250 map-side flush v2 */'
    a2 = "int kfd_bc250_flush_by_runlist(struct kfd_node *dev)\n{"
    r2 = "int kfd_bc250_flush_by_runlist(struct kfd_node *dev, int site)\n{"
    a3 = "\tif (!bc250_flush_by_runlist || !dev || !dev->adev || !dev->adev->pdev)"
    r3 = "\tif (!(bc250_flush_by_runlist & site) || !dev || !dev->adev || !dev->adev->pdev)"
    if a1 not in s or a2 not in s or a3 not in s: return None
    return s.replace(a1, r1).replace(a2, r2).replace(a3, r3)
patch(os.path.join(S, "kfd_device_queue_manager.c"), dqm_tx)

# 2. header declaration
def h_tx(s):
    a = "int kfd_bc250_flush_by_runlist(struct kfd_node *dev);"
    if a not in s: return None
    return s.replace(a, "int kfd_bc250_flush_by_runlist(struct kfd_node *dev, int site); /* bc250 map-side flush v2 */")
patch(os.path.join(S, "kfd_device_queue_manager.h"), h_tx)

# 3. call sites: update unmap site to site=1; add map site (site=2)
def c_tx(s):
    a1 = "\t\tkfd_bc250_flush_by_runlist(peer_pdd->dev);"
    r1 = "\t\tkfd_bc250_flush_by_runlist(peer_pdd->dev, 1); /* bc250 map-side flush v2 */"
    # map ioctl: loop body has the UNconditional kfd_flush_tlb followed by
    # kfree(devices_arr) and "return err;" then the map ioctl's error labels
    a2 = ("\t\tkfd_flush_tlb(peer_pdd);\n"
          "\t}\n"
          "\tkfree(devices_arr);\n"
          "\n"
          "\treturn err;\n"
          "\n"
          "get_process_device_data_failed:")
    r2 = ("\t\tkfd_flush_tlb(peer_pdd);\n"
          "\n"
          "\t\t/* BC-250: the map-side TLB flush above is the broken PASID sweep;\n"
          "\t\t * stale-invalid UTCL2 entries cached while the range was unmapped\n"
          "\t\t * survive it and fault on first touch of the fresh mapping\n"
          "\t\t * (measured: fault 44-122 us after the PTE write). Rebuild the\n"
          "\t\t * runlist here too. Site bit 2.\n"
          "\t\t */\n"
          "\t\tkfd_bc250_flush_by_runlist(peer_pdd->dev, 2);\n"
          "\t}\n"
          "\tkfree(devices_arr);\n"
          "\n"
          "\treturn err;\n"
          "\n"
          "get_process_device_data_failed:")
    if a1 not in s or a2 not in s: return None
    return s.replace(a1, r1).replace(a2, r2, 1)
patch(os.path.join(S, "kfd_chardev.c"), c_tx)

print("all done")
