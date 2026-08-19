#!/usr/bin/env bash
# Loose ends left after the first round.
#
# 1. The quantized-value control was run against the wrong model. qwen3-14B
#    carries a separate known corruption, so it cannot test "models whose value
#    weights are quantized never reach this path". deepseek-r1-14B (q6_K value
#    weights) is the model that claim is about.
# 2. The 8B decode variance was attributed to running near the memory ceiling,
#    but that was an inference from utilisation figures. If it is right, a model
#    far from the ceiling should be stable at depth. The 1.5B sits at about 29
#    percent against the 8B's 80.
# 3. The fp16 defect was tested with graph capture on and off. It has not been
#    tested with flash attention off, which changes the graph shape entirely.
set -u
D=~/loose-end-controls; mkdir -p "$D"
HIP=~/llama-master/build-hip/bin
L=/home/akandr/rocBLAS/build/release/rocblas-install/lib
WIKI=~/wiki.test.raw
DS14=/opt/models/deepseek-r1-14b.gguf
Q15=/opt/models/qwen2.5-1.5b-q4km.gguf
M8=/opt/models/qwen3-8b-q8_0.gguf
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "=== 1. deepseek-r1-14B with fp16 compute: the correct quantized-value control"
for i in 1 2 3 4; do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-perplexity -m $DS14 --no-mmap -ngl 99 -fa on -c 2048 \
    -f $WIKI --chunks 2 > "$D/ds14_fp16_$i.log" 2>&1
  log "  ds14 fp16 run $i: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/ds14_fp16_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done
for i in 1 2; do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-perplexity -m $DS14 --no-mmap -ngl 99 -fa on -c 2048 \
    -f $WIKI --chunks 2 > "$D/ds14_f32_$i.log" 2>&1
  log "  ds14 f32  run $i: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/ds14_f32_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done

log "=== 2. 1.5B decode at depth, ten runs: does a model far from the memory ceiling stay stable?"
for i in $(seq 1 10); do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-bench -m $Q15 -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 \
    > "$D/q15_d_$i.log" 2>&1
  log "  q15 d16128 run $i: $(grep -aoE 'tg8 @ d[0-9]+ \| +[0-9]+\.[0-9]+' "$D/q15_d_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done

log "=== 3. the fp16 defect with flash attention OFF (a different graph shape)"
for i in 1 2 3 4; do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 LD_LIBRARY_PATH=$L \
    timeout -k 20 1800 $HIP/llama-perplexity -m $M8 --no-mmap -ngl 99 -fa off -c 2048 \
    -f $WIKI --chunks 2 > "$D/8b_faoff_$i.log" 2>&1
  log "  8b fp16 fa-off run $i: $(grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/8b_faoff_$i.log" | grep -oE '[0-9.]+$' || echo FAIL)"
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE 'memory access fault|preemption time out')"
touch "$D/DONE"; log done
