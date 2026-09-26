#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Compile-only search for RDNA1 flash-attention tile rows that do not spill, one head size at a
# time. Patches candidate rows into the RDNA1 table (ggml_cuda_fattn_tile_get_config_amd_rdna1),
# compiles just that template instance with -Rpass-analysis=kernel-resource-usage, and prints
# every flash_attn_tile instance's VGPR/spill/scratch. Restores the tree afterwards.
#
# CPU-only, but it heats the APU package: never run this while a GPU measurement is in progress.
#
# Usage: fa_row_sweep.sh <D> "<ncols:nthreads:occ:nbatch_fa:nbatch_K> ..." ["<second candidate set>" ...]
set -u
D=$1; shift
cd ~/llama-master || exit 1
f=ggml/src/ggml-cuda/fattn-tile.cuh
BASE=/tmp/fattn-tile.cuh.patched4
[ -s "$BASE" ] || { echo "no $BASE"; exit 1; }
FL="-O3 -std=c++17 -DGGML_USE_HIP -DGGML_HIP_NO_VMM -I ggml/include -I ggml/src -I ggml/src/ggml-cuda -Rpass-analysis=kernel-resource-usage"
SRC=ggml/src/ggml-cuda/template-instances/fattn-tile-instance-dkq$D-dv$D.cu
[ -s "$SRC" ] || { echo "no $SRC"; exit 1; }

for spec in "$@"; do
  cp "$BASE" "$f"
  python3 - "$D" $spec <<'PY'
import sys
D=sys.argv[1]
p="ggml/src/ggml-cuda/fattn-tile.cuh"; s=open(p).read()
i=s.index("ggml_cuda_fattn_tile_get_config_amd_rdna1")
j=s.index("return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);", i)
import re
# the table is first-match-wins, so a row that already exists for this D and ncols must be
# replaced in place, not shadowed by an insertion after it
for x in sys.argv[2:]:
    ncols = x.split(":")[0]
    line = "    GGML_CUDA_FATTN_TILE_CONFIG_CASE(%s, %s, %s, %s, %s, %s, %s)\n" % ((D, D) + tuple(x.split(":")))
    pat = re.compile(r"    GGML_CUDA_FATTN_TILE_CONFIG_CASE\(%s, %s,\s*%s,[^)]*\)\n" % (D, D, ncols))
    m = pat.search(s, i, j)
    if m:
        s = s[:m.start()] + line + s[m.end():]; j = s.index("return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);", i)
    else:
        s = s[:j] + line + s[j:]; j += len(line)
open(p, "w").write(s)
PY
  echo "== D=$D rows: $spec"
  out=$(hipcc --offload-arch=gfx1013 $FL -c "$SRC" -o /tmp/rowsweep.o 2>&1)
  if echo "$out" | grep -q "error:"; then
    echo "   COMPILE ERROR: $(echo "$out" | grep -m1 'error:' | cut -c1-100)"; continue
  fi
  echo "$out" | sed 's/.*remark: *//; s/ \[-Rpass.*//' \
    | awk '/Function Name: _ZL15flash_attn_tile/{n=$3} /VGPRs:/{v=$2} /ScratchSize/{s=$3} /Occupancy/{o=$3}
           /VGPRs Spill:/{if(n!=""){printf "   %-14s vgpr=%-4s spill=%-5s scratch=%-5s occ=%s\n", n, v, $3, s, o; n=""}}' \
    | sed -E 's/_ZL15flash_attn_tileILi[0-9]+ELi[0-9]+ELi([0-9]+)ELi([0-9]+)ELb([01])E[^ ]*/c1=\1 c2=\2 oob=\3/' \
    | sort -t= -k6 -n -r
done
cp "$BASE" "$f"; echo "restored"
