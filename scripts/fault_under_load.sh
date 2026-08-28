#!/usr/bin/env bash
# Provoke a GPU page fault while queues are busy.
#
# A deliberate out-of-bounds write reproduces the fault signature of the events
# that kill this board, TCP client, PERMISSION_FAULTS 0x5, a write. On an idle
# GPU it goes no further: the process dies and the driver never tries to reset,
# at either setting of amdgpu.gpu_recovery. The fatal events all happened under
# sustained load, where the fault leads to a queue preemption that times out,
# then the runlist rebuild failing with -62, and only then a reset. So the load
# is part of the reproducer, and without it the mitigation cannot be tested
# because the path it guards is never taken.
set -u
D=~/inv113; L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
PAT='GPU reset begin'
PAT2='Queue preemption failed|runlist rebuild flush failed'
log () { echo "[$(date +%T)] $*" >> "$D/underload.log"; sync; }
log "=== arm $1, gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery), uptime $(cut -d. -f1 /proc/uptime)s"
env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH="$L" \
  timeout -k 20 600 ~/llama-master/build-hip/bin/llama-bench -m /opt/models/qwen2.5-1.5b-q4km.gguf \
  -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 3 > "$D/load_$1.log" 2>&1 &
lp=$!
sleep 40
log "  firing the fault probe while the benchmark runs"
timeout -k 10 90 "$D/faultprobe" 1073741824 > "$D/probe_$1.out" 2>&1
log "  probe exit=$?"
wait $lp 2>/dev/null
log "  benchmark: $(grep -aoE 'tg8 @ d[0-9]+ \| +[0-9.]+' "$D/load_$1.log" | tail -1)"
log "  resets this boot: $(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE "$PAT")"
log "  preemption or flush failures: $(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE "$PAT2")"
log "=== done, uptime $(cut -d. -f1 /proc/uptime)s"
