#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# After the D=256 A/B: (1) the third HIP library on the deep-context command, three gdb runs, expecting a
# clean out-of-memory every time; (2) experiment K: D=128 rows for 4 and 2 columns on top of build-hip-fa3's
# state (row32) as build-hip-fa4, one-token FA of the 8B and 1.5B, tg32 at depth 4096 and 0, two passes
# against build-hip-fa3.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/after-d256; H=$HOME/clr-src/hiplib3; R=~/rocr-repro
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -qE "CHAIN DONE|CHAIN FAILED" ~/fa-d256-2/log 2>/dev/null; do sleep 60; done
until grep -q "CLR3 DONE" ~/clr-src/log3 2>/dev/null; do sleep 60; done
sleep 60
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
for i in 1 2 3; do
  LD_LIBRARY_PATH=$H:$L timeout -k 30 1500 gdb -q -batch -x $R/gdb.cmds --args $M/build-hip-f32iq/bin/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 > $R/clrfix3-r$i.log 2>&1
  log "hip patch 2, run $i: $(grep -aoE 'received signal [A-Z]+|ROCm error: [a-z ]+' $R/clrfix3-r$i.log | head -2 | tr '\n' ' ')"
done
# experiment K
cd $M || exit 1
cp /tmp/fattn-tile.cuh.fa3 ggml/src/ggml-cuda/fattn-tile.cuh && python3 ~/apply_rdna1_fa_d128_small.py ggml/src/ggml-cuda | tee -a $O/log
cmake -S . -B build-hip-fa4 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1
nice -n 5 cmake --build build-hip-fa4 -j 7 --target test-backend-ops llama-bench > $O/build.log 2>&1 && log "build fa4 ok" || { log "build FAILED"; grep -m3 -A2 "error:" $O/build.log | tee -a $O/log; }
cp ggml/src/ggml-cuda/fattn-tile.cuh /tmp/fattn-tile.cuh.fa4
cp /tmp/fattn-tile.cuh.patched4 ggml/src/ggml-cuda/fattn-tile.cuh
sleep 120
LD_LIBRARY_PATH=$L timeout -k 30 1800 $M/build-hip-fa4/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 2>&1 | grep -a "FLASH_ATTN_EXT" > $O/tbo-fa4.log
log "fa4 FLASH_ATTN_EXT correctness: OK $(grep -ac 'OK' $O/tbo-fa4.log) FAIL $(grep -ac 'FAIL' $O/tbo-fa4.log)"
for b in build-hip-fa3 build-hip-fa4; do for m in qwen3-8b-q8_0 qwen2.5-1.5b-q4km; do
  LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$b/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b ROCm0 > $O/fa-$b-$m.log 2>&1
  log "$b $m FA us: $(grep -ao '[0-9.]* us/run' $O/fa-$b-$m.log | tr '\n' ' ')"
done; done
tgd() { LD_LIBRARY_PATH=$L timeout -k 20 1500 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 32 -d $3 -r 3 2>/dev/null | grep -a "tg32" | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do for m in qwen3-8b-q8_0 qwen2.5-1.5b-q4km; do
  log "p$pass $m tg32 d=4096: fa3 $(tgd build-hip-fa3 $m 4096) | fa4 $(tgd build-hip-fa4 $m 4096)   d=0: fa3 $(tgd build-hip-fa3 $m 0) | fa4 $(tgd build-hip-fa4 $m 0)"
done; done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
