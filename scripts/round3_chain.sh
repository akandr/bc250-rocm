#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Round three: GPU_MAX_HW_QUEUES across the models, the 8B against the three-patch build, and the D=256
# row 256:2:64:32 (27B 16x2 instance 225 -> 6 spilled registers) as a variant.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/round3; F=$M/build-hip-final/bin
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
cd $M || exit 1
# CPU: variant build with the new row32
cp ggml/src/ggml-cuda/fattn-tile.cuh /tmp/fattn-tile.cuh.seven
python3 - <<'PY'
import re,pathlib
p=pathlib.Path("/home/akandr/llama-master/ggml/src/ggml-cuda/fattn-tile.cuh"); s=p.read_text()
i=s.index("ggml_cuda_fattn_tile_get_config_amd_rdna1"); j=s.index("return ggml_cuda_fattn_tile_get_config_amd_rdna(DKQ, DV, ncols);", i)
m=re.compile(r"    GGML_CUDA_FATTN_TILE_CONFIG_CASE\(256, 256, 32,[^)]*\)\n").search(s,i,j)
s=s[:m.start()]+"    GGML_CUDA_FATTN_TILE_CONFIG_CASE(256, 256, 32, 256, 2, 64, 32)\n"+s[m.end():]; p.write_text(s); print("row32 -> 256,2,64,32")
PY
cmake -S . -B build-hip-fa5 -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1
nice -n 5 cmake --build build-hip-fa5 -j 7 --target llama-bench test-backend-ops > $O/build-fa5.log 2>&1 && log "build fa5 ok" || log "build fa5 FAILED"
cp /tmp/fattn-tile.cuh.seven ggml/src/ggml-cuda/fattn-tile.cuh
sleep 180
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
# GPU 1: hardware queues
bench() { env $1 LD_LIBRARY_PATH=$L timeout -k 20 1500 $F/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512 -n 64 -r 3 2>/dev/null | grep -a "pp512\|tg64" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 deepseek-r1-14b qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  for k in "X=1" "GPU_MAX_HW_QUEUES=1" "GPU_MAX_HW_QUEUES=2"; do log "p$pass $m [$k]: $(bench "$k" $m)"; done
done; done
gate() { env $1 LD_LIBRARY_PATH=$L timeout -k 30 2400 $F/llama-perplexity -m /opt/models/$2.gguf --no-mmap -ngl 99 -fa on -c $3 -f ~/wiki.test.raw --chunks $4 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*'; }
log "1.5B gate, GPU_MAX_HW_QUEUES=1: $(gate GPU_MAX_HW_QUEUES=1 qwen2.5-1.5b-q4km 4096 8)  (8.9498 without)"
log "8B gate, GPU_MAX_HW_QUEUES=1: $(gate GPU_MAX_HW_QUEUES=1 qwen3-8b-q8_0 2048 2)  (9.1273 without)"
KERNTRACE_OUT=$O/trace-1.5b-q1.txt GPU_MAX_HW_QUEUES=1 LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $F/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 1 > /dev/null 2>&1
log "gaps 1.5B with one HW queue: $(head -1 $O/trace-1.5b-q1.txt | cut -c1-120) | $(grep -a 'gaps between' $O/trace-1.5b-q1.txt | cut -c1-230)"
# GPU 2: 8B, three-patch build against the final, interleaved
tg8() { LD_LIBRARY_PATH=$L timeout -k 20 900 $M/$1/bin/llama-bench -m /opt/models/qwen3-8b-q8_0.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 3 2>/dev/null | grep -a tg64 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2 3; do log "p$pass 8B tg64: three-patch (build-hip-f44) $(tg8 build-hip-f44) | final $(tg8 build-hip-final)"; done
# GPU 3: D=256 row variant
for b in build-hip-final build-hip-fa5; do for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$b/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-fa.txt -b ROCm0 > $O/fa-$b-$m.log 2>&1
  log "$b $m FA us: $(grep -ao '[0-9.]* us/run' $O/fa-$b-$m.log | tr '\n' ' ')"
done; done
LD_LIBRARY_PATH=$L timeout -k 30 1800 $M/build-hip-fa5/bin/test-backend-ops test -o FLASH_ATTN_EXT -b ROCm0 2>&1 | grep -a "FLASH_ATTN_EXT" > $O/tbo-fa5.log
log "fa5 FLASH_ATTN_EXT: OK $(grep -ac 'OK' $O/tbo-fa5.log) FAIL $(grep -ac 'FAIL' $O/tbo-fa5.log)"
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512,2048 -n 0 -r 3 2>/dev/null | grep -aE "pp512|pp2048" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do for m in qwen3.6-35b-a3b-iq2m qwen3.8-27b-iq3xxs; do
  log "p$pass $m final: $(pp build-hip-final $m)"; log "p$pass $m fa5 (row32 256:2:64:32): $(pp build-hip-fa5 $m)"
done; done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
