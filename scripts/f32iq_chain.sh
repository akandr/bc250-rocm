#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment G: the float matvec extended to q5_K and the IQ types (iq2_xxs, iq3_xxs, iq3_s, iq4_xs) and to
# MUL_MAT_ID with one token (the MoE's experts). Waits for the GPU queue (perf3, gdb repeats), builds
# build-hip-f32iq, then: correctness against the CPU for MUL_MAT and MUL_MAT_ID, tg128 A/B of the
# previous float build against it on the 27B, the MoE and the 1.5B (control), greedy text on both, and the
# n=1 shapes of the 27B and MoE graphs replayed on both builds and Vulkan.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/f32iq-chain; BASE=build-hip-f32mv; NEW=build-hip-f32iq
mkdir -p $O ~/opgraph
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until [ -f ~/f32all-chain/perf3.done ] && grep -q "REPEAT DONE" ~/rocr-repro/repeat.log 2>/dev/null; do sleep 60; done
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
git diff --stat ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/mmvq-rdna1-f32.cu | tee -a $O/log
cmake -S . -B $NEW -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1 && log "configured" || { log "configure FAILED"; exit 1; }
log "build start"
nice -n 5 cmake --build $NEW -j 7 --target test-backend-ops llama-bench llama-cli test-export-graph-ops > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m5 -B2 -A3 "error:" $O/build.log | tee -a $O/log; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 150
T='type_a=(q4_K|q5_K|q6_K|q8_0|iq2_xxs|iq3_xxs|iq3_s|iq4_xs),type_b=f32'
LD_LIBRARY_PATH=$L timeout -k 30 1800 $M/$NEW/bin/test-backend-ops test -o MUL_MAT -b ROCm0 2>&1 | grep -aE "$T,m=[0-9]+,n=1," > $O/tbo-mm.log
log "MUL_MAT n=1 correctness: OK $(grep -ac 'OK' $O/tbo-mm.log) FAIL $(grep -ac 'FAIL' $O/tbo-mm.log)"
LD_LIBRARY_PATH=$L timeout -k 30 1800 $M/$NEW/bin/test-backend-ops test -o MUL_MAT_ID -b ROCm0 2>&1 | grep -aE "$T" > $O/tbo-mmid.log
log "MUL_MAT_ID correctness: OK $(grep -ac 'OK' $O/tbo-mmid.log) FAIL $(grep -ac 'FAIL' $O/tbo-mmid.log)"
grep -a FAIL $O/tbo-mm.log $O/tbo-mmid.log | head -20 | cut -c1-200 | tee -a $O/log
tg() { LD_LIBRARY_PATH=$L timeout -k 20 1500 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 128 -r 3 2>/dev/null | grep -a tg128 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do
  for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km; do
    for b in $BASE $NEW; do log "p$pass $m $b tg128: $(tg $b $m)"; done
  done
done
for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
  for b in $BASE $NEW; do
    LD_LIBRARY_PATH=$L timeout -k 10 900 $M/$b/bin/llama-cli -m /opt/models/$m.gguf -ngl 99 -fa on -c 4096 -n 64 --temp 0 -no-cnv -st -p 'The three primary colours are' < /dev/null > $O/text-$m-$b.txt 2>&1
  done
  if diff -q <(grep -a -A6 'primary colours are' $O/text-$m-$BASE.txt) <(grep -a -A6 'primary colours are' $O/text-$m-$NEW.txt) > /dev/null; then log "$m greedy text IDENTICAL"; else log "$m greedy text DIFFERS"; fi
done
for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
  OPS=~/opgraph/ops-$m.txt
  [ -s $OPS ] || LD_LIBRARY_PATH=$L $M/$NEW/bin/test-export-graph-ops -m /opt/models/$m.gguf -ngl 99 -c 4096 -b 2048 -ub 2048 -fa off -o $OPS > $O/export-$m.log 2>&1
  log "$m ops exported: $(wc -l < $OPS)"
  for b in $BASE $NEW; do
    LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/$b/bin/test-backend-ops perf --test-file $OPS -b ROCm0 > $O/ops-$m-$b.log 2>&1
    log "$m $b n=1 ops: $(grep -a 'us/run' $O/ops-$m-$b.log | grep -a 'ne=\[[0-9]*,1,[0-9]*,1\]' | grep -ao '[0-9.]* us/run' | tr '\n' ' ')"
  done
  timeout -k 30 2400 $M/build-vk-f44/bin/test-backend-ops perf --test-file $OPS -b Vulkan0 > $O/ops-$m-vk.log 2>&1
  log "$m vulkan n=1 ops: $(grep -a 'us/run' $O/ops-$m-vk.log | grep -a 'ne=\[[0-9]*,1,[0-9]*,1\]' | grep -ao '[0-9.]* us/run' | tr '\n' ' ')"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
