#!/usr/bin/env python3
"""v3: add the BC-250 runlist-rebuild flush to the SVM map/unmap paths.

Function profiling during a faulting TBO churn run: 43 svm_range_validate_and_map
and 14 svm_range_unmap_from_gpus calls, zero restore-worker calls, ioctl paths
already protected (67 rebuilds) yet 3/3 faults. The SVM paths call the broken
kfd_flush_tlb and get no rebuild; the faulting VAs are SVM ranges. Hook both
sites with the existing helper (site 1 = unmap, site 2 = map).
Run on the tree already carrying runlist v1 + mapflush v2.
"""
import sys, os

S = os.path.expanduser(sys.argv[1] if len(sys.argv) > 1
                       else "~/k715/linux-7.1.5/drivers/gpu/drm/amd/amdkfd")
MARK = "bc250 svm flush v3"
path = os.path.join(S, "kfd_svm.c")

with open(path) as f:
    src = f.read()
if MARK in src:
    print("already patched"); sys.exit(0)

# site 1: svm_range_unmap_from_gpus loop, after its kfd_flush_tlb
a1 = ("\t\t\tif (r)\n"
      "\t\t\t\tbreak;\n"
      "\t\t}\n"
      "\t\tkfd_flush_tlb(pdd);\n"
      "\t}\n"
      "\n"
      "\treturn r;\n"
      "}\n"
      "\n"
      "static int\n"
      "svm_range_map_to_gpu(")
r1 = ("\t\t\tif (r)\n"
      "\t\t\t\tbreak;\n"
      "\t\t}\n"
      "\t\tkfd_flush_tlb(pdd);\n"
      "\t\t/* bc250 svm flush v3: SVM unmap needs the rebuild too */\n"
      "\t\tkfd_bc250_flush_by_runlist(pdd->dev, 1);\n"
      "\t}\n"
      "\n"
      "\treturn r;\n"
      "}\n"
      "\n"
      "static int\n"
      "svm_range_map_to_gpu(")

# site 2: svm_range_validate_and_map's map loop, after its kfd_flush_tlb
a2 = ("\t\t\t\tbreak;\n"
      "\t\t\t}\n"
      "\t\t}\n"
      "\n"
      "\t\tkfd_flush_tlb(pdd);\n"
      "\t}\n"
      "\n"
      "\treturn r;\n"
      "}\n"
      "\n"
      "struct svm_validate_context {")
r2 = ("\t\t\t\tbreak;\n"
      "\t\t\t}\n"
      "\t\t}\n"
      "\n"
      "\t\tkfd_flush_tlb(pdd);\n"
      "\t\t/* bc250 svm flush v3: stale-invalid UTCL2 entries survive the\n"
      "\t\t * broken PASID sweep; fault measured 44-122 us after the PTE\n"
      "\t\t * write on remap. Rebuild the runlist on SVM map as well. */\n"
      "\t\tkfd_bc250_flush_by_runlist(pdd->dev, 2);\n"
      "\t}\n"
      "\n"
      "\treturn r;\n"
      "}\n"
      "\n"
      "struct svm_validate_context {")

if a1 not in src: print("FAILED anchor 1"); sys.exit(1)
if a2 not in src: print("FAILED anchor 2"); sys.exit(1)
with open(path + ".pre-svmflush", "w") as f:
    f.write(src)
src = src.replace(a1, r1, 1).replace(a2, r2, 1)
# the helper decl lives in kfd_device_queue_manager.h; kfd_svm.c includes kfd_priv.h
# -> ensure the header is included (kfd_svm.c includes kfd_device_queue_manager.h? check)
if '#include "kfd_device_queue_manager.h"' not in src:
    src = src.replace('#include "kfd_svm.h"', '#include "kfd_svm.h"\n#include "kfd_device_queue_manager.h"', 1)
with open(path, "w") as f:
    f.write(src)
print("patched kfd_svm.c")
