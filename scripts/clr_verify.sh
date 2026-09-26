#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The rebuilt HIP runtime (HostQueue::Thread::Release null guard) on the deep-context command that faulted
# six times of six under gdb with the installed ROCr: three gdb runs with the new libamdhip64 through
# LD_LIBRARY_PATH (not installed), expecting the clean out-of-memory abort every time; then the 1.5B gate
# and pp512/tg64 under the new library against the stock one. Then the D=128 small-column tile row sweep
# (CPU only). Waits for the vec-final chain.
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/rocr-repro; H=$HOME/clr-src/hiplib
log() { echo "[$(date +%T)] $*" | tee -a $O/clrverify.log; }
until grep -qE "CHAIN DONE|CHAIN FAILED" ~/fa-vec-final/log 2>/dev/null; do sleep 60; done
sleep 60
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
: > $O/clrverify.log
log "ldd with hiplib: $(LD_LIBRARY_PATH=$H:$L ldd $M/build-hip-f32iq/bin/llama-bench | grep -a "amdhip64\|hsa-runtime" | tr -s ' ' | tr '\n' ';')"
for i in 1 2 3; do
  LD_LIBRARY_PATH=$H:$L timeout -k 30 1500 gdb -q -batch -x $O/gdb.cmds --args $M/build-hip-f32iq/bin/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 > $O/clrfix-r$i.log 2>&1
  log "run $i: $(grep -aoE 'received signal [A-Z]+|ROCm error: [a-z ]+' $O/clrfix-r$i.log | head -2 | tr '\n' ' ')"
done
gate() { LD_LIBRARY_PATH=$1 timeout -k 30 2400 $M/build-hip-f32mv/bin/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*'; }
tp() { LD_LIBRARY_PATH=$1 timeout -k 20 900 $M/build-hip-f32iq/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 512 -n 64 -r 3 2>/dev/null | grep -a "pp512\|tg64" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
log "1.5B gate, new HIP: $(gate $H:$L)   stock HIP: $(gate $L)   (expect 8.9498)"
for p in 1 2; do log "1.5B p$p new HIP: $(tp $H:$L) | stock: $(tp $L)"; done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
echo "CLRVERIFY DONE" >> $O/clrverify.log
cd $M && ~/fa_row_sweep.sh 128 4:128:4:32:64 4:256:4:32:64 4:128:3:32:32 4:256:2:32:64 4:64:4:32:64 4:128:2:32:128 > ~/fa-remainder/sweep128-c4.log 2>&1
~/fa_row_sweep.sh 128 2:64:4:32:64 2:128:4:32:64 2:64:3:32:32 2:128:2:32:64 > ~/fa-remainder/sweep128-c2.log 2>&1
echo SWEEP128 DONE >> ~/fa-remainder/sweep128-c2.log
