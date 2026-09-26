#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Allocation-churn A/B/A of the runlist flush on Fedora 44 + ROCm 7.1.1 (same kernel module as Fedora 43).
# A: param 3 (unmap+map rebuild), B: param 1 (unmap only, expected to fault), A2: param 3 again.
export PATH=/usr/bin:/usr/sbin HSA_ENABLE_SDMA=0
export LD_LIBRARY_PATH=$HOME/rb711/comgr-fixed:$HOME/rb711/install/lib64
D=~/s0915/churnf44; mkdir -p $D
H=~/llama-master/build-hip-f44/bin; P=/sys/module/amdgpu/parameters/bc250_flush_by_runlist
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults() { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a $D/log; sync; }
phase() { tag=$1 n=$2
  f0=$(faults)
  for i in $(seq 1 $n); do
    timeout -k 10 900 $H/test-backend-ops perf -o MUL_MAT -b ROCm0 > $D/tbo_${tag}_$i.log 2>&1; rc=$?
    rf=$(grep -aciE "memory access fault|VIOLATION" $D/tbo_${tag}_$i.log)
    timeout -k 10 300 ~/s0915/seq_probe_f44 4000 8388608 4194304 8388608 6000000 8388608 > $D/seq_${tag}_$i.log 2>&1; src=$?
    log "  $tag run $i: tbo rc=$rc runtime_faults=$rf | seq rc=$src $(grep -a RESULT $D/seq_${tag}_$i.log)"
  done
  log "PHASE $tag: kernel fault lines +$(( $(faults) - f0 ))"
}
log "boot $(cat /proc/sys/kernel/random/boot_id) $(grep -o "Forty [A-Za-z]*" /etc/fedora-release) param=$(cat $P) faults_at_start=$(faults)"
echo 3 | sudo tee $P >/dev/null; log "=== A param=$(cat $P)"; phase A 3
echo 1 | sudo tee $P >/dev/null; log "=== B param=$(cat $P)"; phase B 2
echo 3 | sudo tee $P >/dev/null; log "=== A2 param=$(cat $P)"; phase A2 3
timeout -k 30 1200 $H/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f ~/wiki.test.raw --chunks 8 > $D/gate_after.log 2>&1
log "gate after A2: $(grep -aoE "Final estimate: PPL = [0-9.]+" $D/gate_after.log)"
log DONE
