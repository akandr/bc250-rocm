#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment J, end to end done right: two build directories. build-hip-gdn holds the current rows (after
# the vec-final rebuild); build-hip-fa3 is the same tree with the D=256 ncols-32 row 256:2:32:64 and the
# 16-column cap off by default. Correctness of FLASH_ATTN_EXT on fa3, the FA lines of both, then
# pp512/pp2048 for the MoE and the 27B, each binary from its own directory, interleaved, two passes.
# Waits for the D=128 sweep (the last CPU job).
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/fa-d256-2
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -q "SWEEP128 DONE" ~/fa-remainder/sweep128-c2.log 2>/dev/null; do sleep 60; done
sleep 30
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
cmp -s ggml/src/ggml-cuda/fattn-tile.cuh /tmp/fattn-tile.cuh.patched4 && log "tree tile table = final rows" || log "WARNING: tree tile table differs from final"
cmake -S . -B build-hip-fa3 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1 || { log "configure FAILED"; exit 1; }
python3 ~/apply_rdna1_fa_d256.py ggml/src/ggml-cuda "512:3:32:128" "256:2:32:64" 0 | tee -a $O/log
nice -n 5 cmake --build build-hip-fa3 -j 7 --target test-backend-ops llama-bench > $O/build.log 2>&1 && log "build fa3 ok" || { log "build FAILED"; grep -m3 -A2 "error:" $O/build.log | tee -a $O/log; cp /tmp/fattn-tile.cuh.patched4 ggml/src/ggml-cuda/fattn-tile.cuh; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
cp ggml/src/ggml-cuda/fattn-tile.cuh /tmp/fattn-tile.cuh.fa3
cp /tmp/fattn-tile.cuh.patched4 ggml/src/ggml-cuda/fattn-tile.cuh     # tree back to the final rows; fa3's binaries keep the row
sleep 120
LD_LIBRARY_PATH=$L timeout -k 30 1800 $M/build-hip-fa3/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 2>&1 | grep -a "FLASH_ATTN_EXT" > $O/tbo-fa3.log
log "fa3 FLASH_ATTN_EXT correctness: OK $(grep -ac 'OK' $O/tbo-fa3.log) FAIL $(grep -ac 'FAIL' $O/tbo-fa3.log)"
for b in build-hip-gdn build-hip-fa3; do for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$b/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b ROCm0 > $O/fa-$b-$m.log 2>&1
  log "$b $m FA us: $(grep -ao '[0-9.]* us/run' $O/fa-$b-$m.log | tr '\n' ' ')"
done; done
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp512|pp2048" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  log "p$pass $m build-hip-gdn (current rows): $(pp build-hip-gdn $m)"
  log "p$pass $m build-hip-fa3 (row32 256:2:32:64): $(pp build-hip-fa3 $m)"
done; done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
