#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Soak on the default Fedora 44 configuration. Each round: qwen2.5-1.5B pp2048 bench, the 1.5B gate
# (reference 8.9442), the qwen3-8B fp16 gate (9.1117, default compute type, exercises rocBLAS fp16 GEMM),
# the MUL_MAT allocation-churn sweep, and every third round the PyTorch training loop if a gfx1013 build is
# present. Faults are counted from the journal of the current boot and the boot before it
# (scripts/fault_count.sh), since dmesg cannot see a fault from a run that ended in a reset.
# Usage: soak_f44.sh [hours] [dir]
set -u
HOURS=${1:-3}; D=${2:-~/soak-f44}; mkdir -p "$D"
HIP=~/llama-master/build-hip-f44/bin
WIKI=~/wiki.test.raw
export HSA_ENABLE_SDMA=0
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }
temp () { sensors 2>/dev/null | grep -oE 'edge:.*\+[0-9.]+' | grep -oE '[0-9.]+$' | head -1; }
end=$(( $(date +%s) + HOURS*3600 )); round=0
# Anything else holding GPU memory invalidates a round: three rounds of the first Fedora 44 soak failed the
# 8B gate because an ollama service loaded a 14B model behind it. Stop other GPU users first; this records
# them so a failed round can be told apart from a board defect.
log "other GPU users at start: ollama $(systemctl is-active ollama 2>/dev/null), processes $(sudo fuser -v /dev/kfd 2>&1 | tail -1)"
log "soak start, ${HOURS}h, boot $(cat /proc/sys/kernel/random/boot_id), faults at start $(faults), previous boot $(sudo journalctl -b -1 -k --no-pager 2>/dev/null | grep -ciE "$PAT")"
while [ "$(date +%s)" -lt "$end" ]; do
  round=$((round+1))
  timeout -k 20 900 $HIP/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -mmp 0 -ngl 99 -fa on -p 2048 -n 0 -r 3 > $D/pp_$round.log 2>&1
  pp=$(grep -aoE "pp2048 +\| +[0-9.]+" $D/pp_$round.log | grep -oE "[0-9.]+$")
  timeout -k 30 1800 $HIP/llama-perplexity -m /opt/models/qwen2.5-1.5b-q4km.gguf --no-mmap -ngl 99 -fa on -c 4096 -f $WIKI --chunks 8 > $D/gate15_$round.log 2>&1
  g1=$(grep -aoE "Final estimate: PPL = [0-9.]+" $D/gate15_$round.log | grep -oE "[0-9.]+$")
  timeout -k 30 1800 $HIP/llama-perplexity -m /opt/models/qwen3-8b-q8_0.gguf --no-mmap -ngl 99 -fa on -c 2048 -f $WIKI --chunks 2 > $D/gate8_$round.log 2>&1
  g2=$(grep -aoE "Final estimate: PPL = [0-9.]+" $D/gate8_$round.log | grep -oE "[0-9.]+$")
  timeout -k 20 1800 $HIP/test-backend-ops perf -o MUL_MAT -b ROCm0 > $D/churn_$round.log 2>&1; crc=$?
  tr=skipped
  if [ $((round % 3)) -eq 0 ] && [ -x ~/torchf44venv/bin/python ]; then
    out=$(timeout -k 20 900 ~/torchf44venv/bin/python ~/s0915/torch_train.py 2>&1)
    tr="$(echo "$out" | grep -a "^cuda" | grep -oE "last loss [0-9.]+") $(echo "$out" | grep -aoE "max loss difference across all steps: [0-9.e+-]+" | sed 's/.*: /maxlossdiff /')"
    echo "$out" | grep -aq "^cuda" || tr=FAILED
  fi
  [ -z "${g1:-}" ] || [ -z "${g2:-}" ] && log "  a gate produced no value; see $D/gate*_$round.log and check for other GPU users"
  log "round=$round pp2048=${pp:-FAIL} gate15=${g1:-FAIL} gate8=${g2:-FAIL} churn_rc=$crc torch='$tr' edge=$(temp) faults=$(faults)"
done
log "summary: rounds=$round distinct gate15: $(grep -oE 'gate15=[^ ]+' $D/log | sort -u | tr '\n' ' ') distinct gate8: $(grep -oE 'gate8=[^ ]+' $D/log | sort -u | tr '\n' ' ') churn failures: $(grep -c 'churn_rc=[^0]' $D/log)"
touch $D/DONE; log done
