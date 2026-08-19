#!/usr/bin/env python3
"""Decode the SDMA queue descriptor from a KFD mqds dump against the kernel struct.

The raw dump is hex words. Reading it by eye invites guessing which word is
which, so the field names here are taken from `struct v10_sdma_mqd` in
drivers/gpu/drm/amd/include/v10_structs.h rather than inferred.

This exists because a coarser reading of the same dumps, counting how many lines
changed between samples, produced a misleading comparison: below the copy-size
threshold ROCclr uses a blit compute kernel and no SDMA queue is created at all,
so the descriptor that appeared to "track live work" was a compute queue, not the
SDMA one. Decoding the fields answers the actual question instead.

Usage: decode_sdma_mqd.py <mqds-dump>
"""
import sys

FIELDS = ["rb_cntl","rb_base","rb_base_hi","rb_rptr","rb_rptr_hi","rb_wptr","rb_wptr_hi",
"rb_wptr_poll_cntl","rb_rptr_addr_hi","rb_rptr_addr_lo","ib_cntl","ib_rptr","ib_offset",
"ib_base_lo","ib_base_hi","ib_size","skip_cntl","context_status","doorbell","status",
"doorbell_log","watermark","doorbell_offset","csa_addr_lo","csa_addr_hi","ib_sub_remain",
"preempt","dummy_reg","rb_wptr_poll_addr_hi","rb_wptr_poll_addr_lo","rb_aql_cntl",
"minor_ptr_update"]

def sdma_words(path):
    out, insec = [], False
    for line in open(path):
        if "SDMA queue" in line:
            insec = True
            continue
        if insec and "queue on device" in line:
            break
        if insec and ":" in line:
            out += line.split(":", 1)[1].split()
    return out

if len(sys.argv) < 2:
    sys.exit(__doc__)
w = sdma_words(sys.argv[1])
if not w:
    sys.exit("no SDMA queue in this dump (below the copy threshold none is created)")
print(f"{len(w)} words in the SDMA queue descriptor\n")
for i, f in enumerate(FIELDS):
    if i < len(w):
        print(f"  {f:<24} 0x{w[i]}")
