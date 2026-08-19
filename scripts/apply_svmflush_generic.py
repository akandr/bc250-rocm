#!/usr/bin/env python3
"""Apply the map-side runlist flush to any amdkfd tree already carrying the v1
(unmap-ioctl) runlist patch. Written after the 7.1.5-specific scripts, because
older trees take a two-argument kfd_flush_tlb and the literal anchors do not
match. Sites are located by enclosing function instead.

Usage: apply_svmflush_generic.py <path-to-drivers/gpu/drm/amd/amdkfd>
"""
import re, sys, os

S = sys.argv[1]
MARK = "bc250 map-side flush (generic)"

def edit(path, fn):
    src = open(path).read()
    if MARK in src:
        print(f"  already patched: {os.path.basename(path)}"); return
    out = fn(src)
    if out is None:
        print(f"  FAILED anchor: {os.path.basename(path)}"); sys.exit(2)
    open(path + ".pre-svmflush-generic", "w").write(src)
    open(path, "w").write(out)
    print(f"  patched: {os.path.basename(path)}")

def body_of(src, fname):
    """Return (start, end) offsets of a function body, matched by name."""
    m = re.search(r"^[\w \t*]*\b" + re.escape(fname) + r"\s*\([^;]*?\)\s*\{", src, re.M | re.S)
    if not m:
        return None
    i = src.index("{", m.start())
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{": depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return (i, j)
    return None

def hook_after_flush_in(src, fname, site, comment):
    """Insert the rebuild call after the kfd_flush_tlb inside fname."""
    span = body_of(src, fname)
    if not span: return None
    start, end = span
    body = src[start:end]
    m = re.search(r"([ \t]*)kfd_flush_tlb\(pdd[^;]*\);", body)
    if not m: return None
    indent = m.group(1)
    ins = (m.group(0) + "\n" +
           f"{indent}/* {comment} ({MARK}) */\n" +
           f"{indent}kfd_bc250_flush_by_runlist(pdd->dev, {site});")
    new_body = body[:m.start()] + ins + body[m.end():]
    return src[:start] + new_body + src[end:]

# 1. helper takes a site mask; parameter becomes a runtime-writable bitmask
def dqm_c(s):
    a1 = 'MODULE_PARM_DESC(bc250_flush_by_runlist,\n\t"BC-250: 1=on unmap, invalidate TLB by rebuilding the runlist (default 0)");'
    r1 = ('MODULE_PARM_DESC(bc250_flush_by_runlist,\n'
          '\t"BC-250: bitmask, rebuild runlist to invalidate TLB: 1=on unmap, 2=on map (default 0)");'
          f' /* {MARK} */')
    a2 = "int kfd_bc250_flush_by_runlist(struct kfd_node *dev)\n{"
    r2 = "int kfd_bc250_flush_by_runlist(struct kfd_node *dev, int site)\n{"
    a3 = "\tif (!bc250_flush_by_runlist || !dev || !dev->adev || !dev->adev->pdev)"
    r3 = "\tif (!(bc250_flush_by_runlist & site) || !dev || !dev->adev || !dev->adev->pdev)"
    a4 = "module_param(bc250_flush_by_runlist, int, 0644);"
    if not all(x in s for x in (a1, a2, a3)): return None
    s = s.replace(a1, r1).replace(a2, r2).replace(a3, r3)
    if a4 not in s:  # v1 shipped it read-only on some trees; make it writable
        s = s.replace("module_param(bc250_flush_by_runlist, int, 0444);", a4)
    return s
edit(os.path.join(S, "kfd_device_queue_manager.c"), dqm_c)

def dqm_h(s):
    a = "int kfd_bc250_flush_by_runlist(struct kfd_node *dev);"
    if a not in s: return None
    return s.replace(a, f"int kfd_bc250_flush_by_runlist(struct kfd_node *dev, int site); /* {MARK} */")
edit(os.path.join(S, "kfd_device_queue_manager.h"), dqm_h)

# 2. ioctl sites: existing unmap call gains site 1, map ioctl gains site 2
def chardev(s):
    a = "\t\tkfd_bc250_flush_by_runlist(peer_pdd->dev);"
    if a not in s: return None
    s = s.replace(a, f"\t\tkfd_bc250_flush_by_runlist(peer_pdd->dev, 1); /* {MARK} */")
    span = body_of(s, "kfd_ioctl_map_memory_to_gpu")
    if not span: return None
    start, end = span
    body = s[start:end]
    m = re.search(r"([ \t]*)kfd_flush_tlb\(peer_pdd[^;]*\);", body)
    if not m: return None
    indent = m.group(1)
    ins = (m.group(0) + "\n" +
           f"{indent}/* BC-250: stale invalid translations survive the broken PASID\n"
           f"{indent} * sweep and fault on first touch of a fresh mapping ({MARK}) */\n" +
           f"{indent}kfd_bc250_flush_by_runlist(peer_pdd->dev, 2);")
    return s[:start] + body[:m.start()] + ins + body[m.end():] + s[end:]
edit(os.path.join(S, "kfd_chardev.c"), chardev)

# 3. the SVM sites, which is where the traffic actually goes
def svm(s):
    if '#include "kfd_device_queue_manager.h"' not in s:
        s = s.replace('#include "kfd_svm.h"', '#include "kfd_svm.h"\n#include "kfd_device_queue_manager.h"', 1)
    s2 = hook_after_flush_in(s, "svm_range_unmap_from_gpus", 1, "BC-250: SVM unmap")
    if s2 is None: return None
    # the map-side flush lives in svm_range_map_to_gpus, not in
    # svm_range_validate_and_map, which only calls it
    s3 = hook_after_flush_in(s2, "svm_range_map_to_gpus", 2,
                             "BC-250: rebuild on SVM map, the case that actually faults")
    return s3
edit(os.path.join(S, "kfd_svm.c"), svm)

print("all done")
