#!/usr/bin/env bash
# Three gaps left open after the loose-end round: whether forward-pass count
# governs the fp16 defect, which ROCclr constant defines the SDMA boundary, and
# whether decode variance tracks memory-bandwidth utilisation.
#
# Note on the first arm: it was designed expecting --chunks to select the number
# of forward passes, which it does not (one chunk at -c 2048 is already several).
# The measurement it produces is still informative, but not as the pass-count
# test it was written as. See the README section on the zeroed fp16 GEMM.
set -u
D=~/inv61; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
WIKI=~/wiki.test.raw
M8=/opt/models/qwen3-8b-q8_0.gguf
Q14=/opt/models/qwen3-14b.gguf
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

# GAP 1. The fp16 defect is deterministic after the first forward pass: pass one
# is clean, every later pass has its first value projection zeroed. If the first
# pass is genuinely special, then a run with only one pass should never show it,
# and more passes should show proportionally more. --chunks controls passes.
log "=== GAP 1: does pass count control the defect? (fp16 compute, 8B)"
for c in 1 2 4; do
  for i in 1 2; do
    env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L \
      timeout -k 20 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
      -f $WIKI --chunks $c > "$D/g1_c${c}_$i.log" 2>&1
    log "  chunks=$c run $i: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/g1_c${c}_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
  done
done
log "  (f32 control at each chunk count, for the reference value)"
for c in 1 2 4; do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
    -f $WIKI --chunks $c > "$D/g1_f32_c$c.log" 2>&1
  log "  f32 chunks=$c: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/g1_f32_c$c.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done

# GAP 2. The SDMA boundary tracks GPU_FORCE_BLIT_COPY_SIZE exactly. Does it also
# move with the staging-buffer knob, which would say which constant defines it?
log "=== GAP 2: does the SDMA threshold move with GPU_STAGING_BUFFER_SIZE?"
for v in unset 4 64 1024; do
  if [ "$v" = unset ]; then r=$(HSA_ENABLE_SDMA=1 timeout -k 5 25 /tmp/h2d 1048576 2>/dev/null | tail -1)
  else r=$(HSA_ENABLE_SDMA=1 GPU_STAGING_BUFFER_SIZE=$v timeout -k 5 25 /tmp/h2d 1048576 2>/dev/null | tail -1); fi
  log "  GPU_STAGING_BUFFER_SIZE=$v -> ${r:-HUNG}"
done
log "  (control: the knob already known to move it)"
for v in 512 1024; do
  r=$(HSA_ENABLE_SDMA=1 GPU_FORCE_BLIT_COPY_SIZE=$v timeout -k 5 25 /tmp/h2d 1048576 2>/dev/null | tail -1)
  log "  GPU_FORCE_BLIT_COPY_SIZE=$v -> ${r:-HUNG}"
done

# GAP 3. Decode variance was shown to differ 6.7x between a model at 80 percent
# of the bandwidth ceiling and one at 29. A model in between says whether that
# scales smoothly or has a threshold. The 14B sits at about 42 percent.
log "=== GAP 3: decode variance for a model at intermediate bandwidth utilisation (14B, ~42 percent)"
for i in $(seq 1 10); do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 GGML_CUDA_DISABLE_GRAPHS=1 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-bench -m $Q14 -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 \
    > "$D/g3_$i.log" 2>&1
  log "  q14 d16128 run $i: $(grep -aoE 'tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+' "$D/g3_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE 'memory access fault|preemption time out')"
touch "$D/DONE"; log done
