#!/bin/bash
# The untested winning combination: NATIVE gfx1013 build (correct apertures) + flash
# attention (-fa on, avoids batched rocBLAS GEMM) + FORCE_MMQ (quantized via ggml native
# kernels). NO HSA_OVERRIDE, NO rocBLAS. On the patched (fixed compute queue) module.
# Runs ON THE BOARD, fresh boot.
set -u
L=~/sweeplogs/native_fa; mkdir -p $L
BIN=$HOME/llama.cpp/build-hip/bin       # native gfx1013 build
M=/opt/models/qwen2.5-1.5b-q4km.gguf
sudo systemctl stop ollama queue-runner 2>/dev/null; sleep 2
echo "=== native_fa $(date +%FT%T)" | tee $L/nfa.log
grep -m1 flush_pasid /usr/src/linux-6.18.9/drivers/gpu/drm/amd/amdgpu/gmc_v10_0.c | tee -a $L/nfa.log

run() { # label flags env
  echo "######## $1" | tee -a $L/nfa.log
  env HSA_ENABLE_SDMA=0 $3 LD_LIBRARY_PATH=$BIN \
    timeout -k 5 200 $BIN/llama-bench -m $M -ngl 999 $2 -r 1 -mmp 0 > $L/${1}.log 2>&1
  echo "$1 RC=$? (124=hang,134=abort,0=OK)" | tee -a $L/nfa.log
  grep -vE "^\[|gdb|debuginfo|LWP|Thread deb|Using host|Enable|Debuginfod|auto-download|<ima|<https|^#[0-9]|warning: The current" $L/${1}.log | tail -14 | tee -a $L/nfa.log
}

# A: native + FA on + FORCE_MMQ, pp128 only (does prompt processing complete?)
run "A_native_fa_mmq_pp" "-fa 1 -p 128 -n 0"  "GGML_CUDA_FORCE_MMQ=1"
# B: native + FA on + FORCE_MMQ, pp128 + tg32 (full)
run "B_native_fa_mmq_full" "-fa 1 -p 128 -n 32" "GGML_CUDA_FORCE_MMQ=1"

echo "=== dmesg:" | tee -a $L/nfa.log
sudo dmesg | grep -iE "amdgpu.*(fault|timeout|preemption|aperture|MEMVIOL)|quiesce|Cannot find CO" | tail -8 | tee -a $L/nfa.log
echo "=== native_fa done $(date +%FT%T)" | tee -a $L/nfa.log
