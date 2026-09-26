#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment H: GATED_DELTA_NET lanes per column on RDNA1 and the transposed-source CONCAT. Builds
# build-hip-gdn from the final tree plus the two kernel changes and the extra test cases; correctness,
# per-op replay of the linear-attention lines of the 27B and MoE graphs at each lane count, pp512/pp2048/
# tg64 A/B against build-hip-f32iq, MoE perplexity on both, and kernel traces with libkerntrace.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/gdn-chain; BASE=build-hip-f32iq; NEW=build-hip-gdn
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
cmake -S . -B $NEW -DGGML_HIP=ON -DAMDGPU_TARGETS=gfx1013 -DCMAKE_BUILD_TYPE=Release -DGGML_HIP_NO_VMM=ON -DLLAMA_CURL=OFF > $O/configure.log 2>&1 && log "configured" || { log "configure FAILED"; exit 1; }
log "build start"
nice -n 5 cmake --build $NEW -j 7 --target test-backend-ops llama-bench llama-cli llama-perplexity > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m5 -B2 -A3 "error:" $O/build.log | tee -a $O/log; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 120
for lanes in 4 8 16 0; do
  GGML_GDN_LANES=$lanes LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$NEW/bin/test-backend-ops test -o GATED_DELTA_NET -b ROCm0 2>&1 | grep -a "GATED_DELTA_NET" > $O/tbo-gdn-l$lanes.log
  log "GDN lanes=$lanes correctness: OK $(grep -ac 'OK' $O/tbo-gdn-l$lanes.log) FAIL $(grep -ac 'FAIL' $O/tbo-gdn-l$lanes.log)"
done
LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$NEW/bin/test-backend-ops test -o CONCAT -b ROCm0 2>&1 | grep -a "CONCAT" > $O/tbo-concat.log
log "CONCAT correctness: OK $(grep -ac 'OK' $O/tbo-concat.log) FAIL $(grep -ac 'FAIL' $O/tbo-concat.log)  (v=16 cases: $(grep -a 'v=16' $O/tbo-concat.log | grep -ac OK) OK, $(grep -a 'v=16' $O/tbo-concat.log | grep -ac FAIL) FAIL)"
grep -a FAIL $O/tbo-concat.log $O/tbo-gdn-l*.log | head -8 | cut -c1-200 | tee -a $O/log
for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
  grep -a "node_44\|conv_input\|conv_output_raw" ~/opgraph/ops-$m.txt > ~/opgraph/ops-$m-gdn.txt
  log "$m linear-attention lines: $(wc -l < ~/opgraph/ops-$m-gdn.txt)"
  for p in 1 2; do
    LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$BASE/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-gdn.txt -b ROCm0 > $O/ops-$m-$BASE-p$p.log 2>&1
    for lanes in 4 8 16 0; do
      GGML_GDN_LANES=$lanes LD_LIBRARY_PATH=$L timeout -k 30 900 $M/$NEW/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-gdn.txt -b ROCm0 > $O/ops-$m-$NEW-l$lanes-p$p.log 2>&1
    done
  done
  timeout -k 30 900 $M/build-vk-f44/bin/test-backend-ops perf --test-file ~/opgraph/ops-$m-gdn.txt -b Vulkan0 > $O/ops-$m-vk.log 2>&1
  log "$m per-op replays done"
done
pp() { LD_LIBRARY_PATH=$L timeout -k 20 1800 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 512,2048 -n 64 -r 3 2>/dev/null | grep -aE "pp512|pp2048|tg64" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
for pass in 1 2; do
  for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
    log "p$pass $m $BASE: $(pp $BASE $m)"
    log "p$pass $m $NEW (lanes 8): $(pp $NEW $m)"
  done
done
for b in $BASE $NEW; do
  log "MoE gate $b: $(LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/$b/bin/llama-perplexity -m /opt/models/qwen3.6-35b-a3b-iq2m.gguf --no-mmap -ngl 99 -fa on -c 2048 -f ~/wiki.test.raw --chunks 3 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*')"
done
for b in $BASE $NEW; do
  LD_LIBRARY_PATH=$L timeout -k 10 900 $M/$b/bin/llama-cli -m /opt/models/qwen3.8-27b-iq3xxs.gguf -ngl 99 -fa on -c 4096 -n 48 --temp 0 -no-cnv -st -p 'The three primary colours are' < /dev/null > $O/text-27b-$b.txt 2>&1
done
if diff -q <(grep -a -A30 'primary colours are' $O/text-27b-$BASE.txt | grep -av 'Prompt:') <(grep -a -A30 'primary colours are' $O/text-27b-$NEW.txt | grep -av 'Prompt:') > /dev/null; then log "27B greedy text IDENTICAL"; else log "27B greedy text DIFFERS"; fi
# kernel traces of real runs on the current front-page build
for m in qwen2.5-1.5b-q4km qwen3.8-27b-iq3xxs; do
  KERNTRACE_OUT=$O/trace-$m-tg.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $M/$BASE/bin/llama-bench -m /opt/models/$m.gguf -ngl 99 -fa 1 -p 0 -n 64 -r 1 > $O/trace-$m-tg.bench 2>&1
  KERNTRACE_OUT=$O/trace-$m-pp.txt LD_PRELOAD=$HOME/libkerntrace.so LD_LIBRARY_PATH=$L timeout -k 20 900 $M/$BASE/bin/llama-bench -m /opt/models/$m.gguf -ngl 99 -fa 1 -p 512 -n 0 -r 1 > $O/trace-$m-pp.bench 2>&1
  log "trace $m: $(head -1 $O/trace-$m-tg.txt 2>/dev/null | cut -c1-120) | pp: $(head -1 $O/trace-$m-pp.txt 2>/dev/null | cut -c1-120)"
done
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
