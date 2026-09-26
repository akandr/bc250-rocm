#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# The patched ROCr (PR #2850 port) turned the deep-context segfault into a clean out-of-memory error in
# seven runs of seven. Before it joins /opt/bc250-rocm/lib64: the 1.5B and 8B perplexity gates and the
# 1.5B's pp512/tg64 under the patched library against the stock one; then install (copy + symlink +
# ldconfig), confirm ld.so prefers it, and run the deep-context command once more with no environment.
# Rollback: sudo rm /opt/bc250-rocm/lib64/libhsa-runtime64.so.1*; sudo ldconfig. Waits for perf5.
set -u
L=/opt/bc250-rocm/lib64; M=~/llama-master; B=$M/build-hip-f32iq/bin; O=~/rocr-repro; P=$HOME/rocr-src/lib
log() { echo "[$(date +%T)] $*" | tee -a $O/validate.log; }
until [ -f ~/f32iq-chain2/perf5/done ]; do sleep 60; done
sleep 60
sudo -n systemctl stop hw-watcher.timer crond 2>/dev/null
: > $O/validate.log
gate() { LD_LIBRARY_PATH=$1 timeout -k 30 2400 $B/llama-perplexity -m /opt/models/$2.gguf --no-mmap -ngl 99 -fa on -c $3 -f ~/wiki.test.raw --chunks $4 2>&1 | grep -ao 'Final estimate: PPL = [0-9.]*'; }
tp() { LD_LIBRARY_PATH=$1 timeout -k 20 900 $B/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa 1 -p 512 -n 64 -r 3 2>/dev/null | grep -a "pp512\|tg64" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr -s ' ' | tr '\n' ';'; }
log "stock   1.5B gate: $(gate $L qwen2.5-1.5b-q4km 4096 8)"
log "patched 1.5B gate: $(gate $P:$L qwen2.5-1.5b-q4km 4096 8)"
log "stock   8B gate: $(gate $L qwen3-8b-q8_0 2048 2)"
log "patched 8B gate: $(gate $P:$L qwen3-8b-q8_0 2048 2)"
for p in 1 2; do
  log "stock   1.5B p$p: $(tp $L)"
  log "patched 1.5B p$p: $(tp $P:$L)"
done
log "ldd before: $(LD_LIBRARY_PATH=$L ldd $B/llama-bench | grep -a hsa-runtime | tr -s ' ')"
sudo -n cp $P/libhsa-runtime64.so.1.18.0 $L/libhsa-runtime64.so.1.18.0 && sudo -n ln -sfn libhsa-runtime64.so.1.18.0 $L/libhsa-runtime64.so.1 && sudo -n ldconfig && log "installed into $L" || { log "INSTALL FAILED"; sudo -n systemctl start hw-watcher.timer crond 2>/dev/null; echo "VALIDATE DONE" >> $O/validate.log; exit 1; }
log "ldconfig: $(ldconfig -p | grep -a 'libhsa-runtime64.so.1 ' | tr -s ' ' | tr '\n' ';')"
log "ldd after: $(ldd $B/llama-bench | grep -a hsa-runtime | tr -s ' ')"
log "md5 installed $(md5sum $L/libhsa-runtime64.so.1.18.0 | cut -c1-12) built $(md5sum $P/libhsa-runtime64.so.1.18.0 | cut -c1-12) stock $(md5sum /usr/lib64/libhsa-runtime64.so.1.18.0 | cut -c1-12)"
timeout -k 30 1500 $B/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 -v > $O/installed-d16384.log 2>&1; rc=$?
log "deep context with the installed runtime: rc=$rc $(grep -aoE 'ROCm error.*' $O/installed-d16384.log | head -1) $(journalctl -k --since '-6min' --no-pager 2>/dev/null | grep -a segfault | tail -1 | grep -oE 'segfault at [0-9a-f]+ .*' | cut -c1-100)"
log "installed 1.5B: $(tp $L)"
sudo -n systemctl start hw-watcher.timer crond 2>/dev/null
echo "VALIDATE DONE" >> $O/validate.log
