#!/usr/bin/env bash
# Which of the three llama.cpp patches does current master still need?
#
# Upstream moved in September 2026: PR #28604 (merged 2026-09-08) hard-codes
# integrated=false on HIP builds, which is what patches/llamacpp/0001 does, and
# #28576 (2026-09-14) switches the flash-attention MMA path to fp32
# accumulation on MFMA devices, which is near the KQV precision problem 0002
# addresses. This runs the gates that caught each defect against an unpatched
# master build, so each patch is kept or retired on a measurement rather than
# on reading the diff.
#
#   gate    1.5B wikitext, ctx 4096, 8 chunks, reference 8.9442 (caught 0001)
#   text    greedy generation, garbled output was the 0003 signature
#   8B      qwen3-8B, ctx 2048, 2 chunks, f32 compute type, reference 9.0975
#   bench   1.5B pp512/tg64, reference 806/113.5
#
# Usage on the board: SRC=~/llama-new LABEL=unpatched ./llamacpp_master_recheck.sh
#
# Fault counting here reads dmesg on a boot that is still up, and was left that way once the run
# was logged, since changing the counter would change what the log means. dmesg cannot see a
# fault from a run that ended by taking the board down; new work should use
# scripts/fault_count.sh, which reads the persistent journal.
set -u
SRC=${SRC:-$HOME/llama-new}
LABEL=${LABEL:-unpatched}
D=${D:-$HOME/s0914/llama-$LABEL}; mkdir -p "$D"
B=$SRC/build-hip/bin
# master replaced --no-mmap with --load-mode (seen 2026-09-14); pick whichever this build accepts
NOMMAP=--no-mmap; "$B/llama-perplexity" --help 2>&1 | grep -q -- "--load-mode" && NOMMAP="-lm none"
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
M8=/opt/models/qwen3-8b-q8_0.gguf
export GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32
export LD_LIBRARY_PATH=/home/akandr/rocBLAS/build/release/rocblas-install/lib
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== $LABEL: $(git -C "$SRC" log -1 --format='%h %s') ; $(git -C "$SRC" status -s | wc -l) modified files"
log "kernel $(uname -r) runlist=$(cat /sys/module/amdgpu/parameters/bc250_flush_by_runlist)"

timeout -k 30 1500 "$B/llama-perplexity" -m $Q15 $NOMMAP -ngl 99 -fa on -c 4096 \
  -f ~/wiki.test.raw --chunks 8 > "$D/gate.log" 2>&1
log "gate 1.5B (ref 8.9442): $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/gate.log" || echo FAIL)"

GEN=$B/llama-completion; [ -x "$GEN" ] || GEN=$B/llama-cli
timeout -k 20 600 "$GEN" -m $Q15 -ngl 99 -fa on -no-cnv -p "The capital of France is" \
  -n 24 --temp 0 > "$D/gen.log" 2>/dev/null </dev/null
log "text: $(tr '\n' ' ' < "$D/gen.log" | cut -c1-200)"

timeout -k 30 1800 "$B/llama-perplexity" -m $M8 $NOMMAP -ngl 99 -fa on -c 2048 \
  -f ~/wiki.test.raw --chunks 2 > "$D/ppl8b.log" 2>&1
log "8B f32 (ref 9.0975): $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/ppl8b.log" || echo FAIL)"

timeout -k 30 900 "$B/llama-bench" -m $Q15 -ngl 99 -fa 1 -p 512 -n 64 > "$D/bench.log" 2>&1
log "bench: $(grep -aE 'pp512|tg64' "$D/bench.log" | awk -F'|' '{print $(NF-2) $(NF-1)}' | tr '\n' ' ')"
log "dmesg faults: $(sudo dmesg | grep -ciE 'page fault|GCVM_L2|preemption time')"
log "=== done"
