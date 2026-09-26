#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Final assembly, 2026-09-19: the tree in its final form (tile rows from J and K, the vec rule, GDN lanes,
# the transposed concat), the patches regenerated from it, the two-patch HIP library checked and installed,
# a fresh build, the split campaign, and tg32 at depth 4096 for every model on ROCm and Vulkan.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/final-assembly; H=$HOME/clr-src/hiplib3
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
# 1. tree
cp /tmp/fattn-tile.cuh.patched4 ggml/src/ggml-cuda/fattn-tile.cuh
python3 ~/apply_rdna1_fa_d256_row.py ggml/src/ggml-cuda | tee -a $O/log
python3 ~/apply_rdna1_fa_d128_small.py ggml/src/ggml-cuda | tee -a $O/log
grep -c "GGML_FA_D256_CAP16\|GGML_FA_VEC_DECODE\|GGML_GDN_LANES" ggml/src/ggml-cuda/fattn-tile.cuh ggml/src/ggml-cuda/fattn.cu ggml/src/ggml-cuda/gated_delta_net.cu | tee -a $O/log
git status --short | tee -a $O/log
# 2. patches
git diff -- ggml/src/ggml-cuda/fattn-tile.cuh ggml/src/ggml-cuda/fattn.cu > $O/0004-rdna1-fattn.patch
git diff -- ggml/src/ggml-cuda/mmvq.cu ggml/src/ggml-cuda/vecdotq.cuh ggml/src/ggml-cuda/mmvq-rdna1-f32.cu ggml/src/ggml-cuda/mmvq-rdna1-f32.cuh > $O/0005-rdna1-mmvq.patch
git diff -- ggml/src/ggml-cuda/concat.cu tests/test-backend-ops.cpp > $O/0006-concat-transposed-source.patch
git diff -- ggml/src/ggml-cuda/gated_delta_net.cu > $O/0007-rdna1-gdn-lanes.patch
wc -l $O/000*.patch | tee -a $O/log
git diff --stat | tail -1 | tee -a $O/log
# 3. HIP library: gate and throughput with both patches, then install
gate() { LD_LIBRARY_PATH=$1 timeout -k 30 2400 $M/build-hip-f32mv/bin/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*'; }
tp() { LD_LIBRARY_PATH=$1 timeout -k 20 900 $M/build-hip-f32iq/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 512 -n 64 -r 3 2>/dev/null | grep -a "pp512\|tg64" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
g=$(gate $H:$L); log "1.5B gate, HIP with both patches: $g  (expect 8.9498)"
log "1.5B tp, HIP both patches: $(tp $H:$L) | stock: $(tp $L)"
if echo "$g" | grep -q "8.9498"; then
  sudo -n install -m 0755 $H/libamdhip64.so.7.1.52802 $L/libamdhip64.so.7.1.52802 && sudo -n ln -sfn libamdhip64.so.7.1.52802 $L/libamdhip64.so.7 && sudo -n ldconfig && log "HIP installed into $L: $(ldconfig -p | grep -a 'libamdhip64.so.7 ' | head -1 | tr -s ' ')"
  timeout -k 30 1500 $M/build-hip-f32iq/bin/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 -v > $O/installed-d16384.log 2>&1; rc=$?
  log "deep context, installed stack, no env: rc=$rc $(grep -aoE 'ROCm error: [a-z ]+' $O/installed-d16384.log | head -1) $(journalctl -k --since '-6min' --no-pager 2>/dev/null | grep -a segfault | tail -1 | grep -oE 'segfault at [0-9a-f]+ .*in lib[a-z0-9_.-]+' | cut -c1-100)"
else
  log "gate mismatch, HIP NOT installed"
fi
# 4. final build
cmake -S . -B build-hip-final -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1 || { log "configure FAILED"; exit 1; }
log "build start"
nice -n 5 cmake --build build-hip-final -j 7 --target test-backend-ops llama-bench llama-cli llama-perplexity > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m3 -A2 "error:" $O/build.log | tee -a $O/log; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 120
B=$M/build-hip-final/bin
for op in FLASH_ATTN_EXT GATED_DELTA_NET CONCAT; do
  LD_LIBRARY_PATH=$L timeout -k 30 1800 $B/test-backend-ops test -o $op -b ROCm0 2>&1 | grep -a "$op" > $O/tbo-$op.log
  log "$op: OK $(grep -ac 'OK' $O/tbo-$op.log) FAIL $(grep -ac 'FAIL' $O/tbo-$op.log)"
done
log "1.5B gate, final build: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
log "8B gate, final build: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $B/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 2 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
sleep 120
HIPBIN=$B ~/campaign_split_f44.sh ~/campaign-final8 > $O/campaign.out 2>&1
log "campaign rc=$? lines=$(wc -l < ~/campaign-final8/log)"
# 5. depth table: tg32 at 4096 and 0, every model, ROCm final and Vulkan
tgd() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $1 -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 32 -d $3 -r 3 2>/dev/null | grep -a "tg32" | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  log "depth $m: rocm d4096 $(tgd $B/llama-bench $m 4096) d0 $(tgd $B/llama-bench $m 0) | vulkan d4096 $(tgd $M/build-vk-f44/bin/llama-bench $m 4096) d0 $(tgd $M/build-vk-f44/bin/llama-bench $m 0)"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
