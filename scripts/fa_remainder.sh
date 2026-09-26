#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The flash-attention remainder on the final build: export every model's graph with -fa on (2048-token
# batch plus the one-token step), keep only the FLASH_ATTN_EXT lines (op id read from ggml.h), replay them
# on build-hip-f32iq and Vulkan, two passes. Waits for the relink series.
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/fa-remainder; mkdir -p $O ~/opgraph
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -q "RELINK DONE" ~/rocr-repro/relink.log 2>/dev/null; do sleep 60; done
sleep 60
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
OPID=$(python3 - <<'PY'
import re
s=open("/home/akandr/llama-master/ggml/include/ggml.h").read()
body=s[s.index("enum ggml_op {"):]; body=body[:body.index("};")]
names=[l.strip().split(",")[0].split("=")[0].strip() for l in body.splitlines()[1:] if l.strip() and not l.strip().startswith("//")]
names=[n for n in names if n.startswith("GGML_OP_")]
print(names.index("GGML_OP_FLASH_ATTN_EXT"))
PY
)
log "GGML_OP_FLASH_ATTN_EXT = $OPID"
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  E=~/opgraph/ops-$m-faon.txt
  [ -s $E ] || LD_LIBRARY_PATH=$L $M/build-hip-f32iq/bin/test-export-graph-ops -m /opt/models/$m.gguf -ngl 99 -c 4096 -b 2048 -ub 2048 -fa on -o $E > $O/export-$m.log 2>&1
  awk -v id=$OPID '$1==id' $E > ~/opgraph/ops-$m-fa.txt
  log "$m: $(wc -l < $E) ops exported, $(wc -l < ~/opgraph/ops-$m-fa.txt) flash-attention lines"
  for p in 1 2; do
    LD_LIBRARY_PATH=$L timeout -k 30 900 $M/build-hip-f32iq/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b ROCm0 > $O/fa-$m-hip-p$p.log 2>&1
    timeout -k 30 900 $M/build-vk-f44/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b Vulkan0 > $O/fa-$m-vk-p$p.log 2>&1
  done
  log "$m: hip $(grep -ao '[0-9.]* us/run' $O/fa-$m-hip-p1.log | tr '\n' ' ') | vk $(grep -ao '[0-9.]* us/run' $O/fa-$m-vk-p1.log | tr '\n' ' ')"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "FA DONE"
