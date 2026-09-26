#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Experiment G, second build: iq4_xs taken out of the float kernel (0.6x on the 27B's dense iq4_xs
# shapes). Waits for the un-debugged crash runs, rebuilds build-hip-f32iq in place, re-checks
# correctness, then tg128 A/B against the q4/q6/q8 float build (27B, MoE, 1.5B control), greedy text on
# the 27B and MoE (stats line excluded), the 27B graph replay, and the full split campaign.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/f32iq-chain2; BASE=build-hip-f32mv; NEW=build-hip-f32iq
mkdir -p $O
log() { echo "[$(date +%T)] $*" | tee -a $O/log; }
until grep -q "NOGDB DONE" ~/rocr-repro/nogdb.log 2>/dev/null; do sleep 60; done
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
cd $M || exit 1
log "build start"
nice -n 5 cmake --build $NEW -j 7 --target test-backend-ops llama-bench llama-cli > $O/build.log 2>&1 && log "build ok" || { log "build FAILED"; grep -m5 -B2 -A3 "error:" $O/build.log | tee -a $O/log; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "CHAIN FAILED" >> $O/log; exit 1; }
sleep 150
T='type_a=(q4_K|q5_K|q6_K|q8_0|iq2_xxs|iq3_xxs|iq3_s|iq4_xs),type_b=f32'
LD_LIBRARY_PATH=$L timeout -k 30 1800 $M/$NEW/bin/test-backend-ops test -o MUL_MAT -b ROCm0 2>&1 | grep -aE "$T,m=[0-9]+,n=1," > $O/tbo-mm.log
log "MUL_MAT n=1 correctness: OK $(grep -ac 'OK' $O/tbo-mm.log) FAIL $(grep -ac 'FAIL' $O/tbo-mm.log)"
LD_LIBRARY_PATH=$L timeout -k 30 1800 $M/$NEW/bin/test-backend-ops test -o MUL_MAT_ID -b ROCm0 2>&1 | grep -aE "$T" > $O/tbo-mmid.log
log "MUL_MAT_ID correctness: OK $(grep -ac 'OK' $O/tbo-mmid.log) FAIL $(grep -ac 'FAIL' $O/tbo-mmid.log)"
tg() { LD_LIBRARY_PATH=$L timeout -k 20 1500 $M/$1/bin/llama-bench -m /opt/models/$2.gguf -ngl 99 -fa 1 -p 0 -n 128 -r 3 2>/dev/null | grep -a tg128 | awk -F'|' '{print $(NF-1)}' | tr -d ' '; }
for pass in 1 2; do
  for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km; do
    for b in $BASE $NEW; do log "p$pass $m $b tg128: $(tg $b $m)"; done
  done
done
for m in qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m; do
  LD_LIBRARY_PATH=$L timeout -k 10 900 $M/$NEW/bin/llama-cli -m /opt/models/$m.gguf -ngl 99 -fa on -c 4096 -n 64 --temp 0 -no-cnv -st -p 'The three primary colours are' < /dev/null > $O/text-$m-$NEW.txt 2>&1
  cp ~/f32iq-chain/text-$m-$BASE.txt $O/
  if diff -q <(grep -a -A40 'primary colours are' $O/text-$m-$BASE.txt | grep -av 'Prompt:') <(grep -a -A40 'primary colours are' $O/text-$m-$NEW.txt | grep -av 'Prompt:') > /dev/null; then log "$m greedy text IDENTICAL"; else log "$m greedy text DIFFERS"; fi
done
OPS=~/opgraph/ops-qwen3.8-27b-iq3xxs.txt
LD_LIBRARY_PATH=$L timeout -k 30 2400 $M/$NEW/bin/test-backend-ops perf --test-file $OPS -b ROCm0 > $O/ops-qwen3.8-27b-iq3xxs-$NEW.log 2>&1
log "27B ops replayed"
sleep 120
HIPBIN=$M/$NEW/bin ~/campaign_split_f44.sh ~/campaign-final7 > $O/campaign.out 2>&1
log "campaign rc=$? lines=$(wc -l < ~/campaign-final7/log)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
log "CHAIN DONE"
