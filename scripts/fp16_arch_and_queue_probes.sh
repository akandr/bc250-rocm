#!/usr/bin/env bash
# Two experiments, both cheap, both aimed at open defects.
#
# A. The fp16 defect disappears when the device is presented as gfx1010 and the
#    system rocBLAS serves the GEMMs. Two things changed at once there: the ISA
#    the kernels target, and which library serves them. The native build carries
#    only gfx1013 code objects so those cannot be separated directly, but a
#    cheaper question can be asked: is it gfx1010 specifically, or does any
#    override do it? The system library also carries gfx1030. If gfx1030 is
#    equally clean, the answer is "anything but the gfx1013 path", which is a
#    different and more useful statement than "gfx1010 fixes it".
#
# B. Decode variance lives in per-process setup. CPU pinning and memory state
#    are both eliminated. Something else differs per process: the hardware queue
#    and doorbell the runtime is given. That is visible in KFD debugfs and has
#    never been recorded alongside the rate.
set -u
exec 9>~/.twoprobes.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv118; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
WIKI=~/wiki.test.raw
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== A. is the fp16 defect specific to gfx1010, or to anything but gfx1013?"
run_ppl () { # run_ppl <tag> <extra env...>
  local tag=$1; shift
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f16 "$@" \
    timeout -k 30 1800 "$HIP/llama-perplexity" -m $M8 --no-mmap -ngl 99 -fa on -c 2048 \
    -f "$WIKI" --chunks 2 > "$D/$tag.log" 2>&1
  grep -aoE 'Final estimate: PPL = [0-9.]+' "$D/$tag.log" | grep -oE '[0-9.]+$' || echo FAIL
}
for i in 1 2 3; do
  a=$(run_ppl "native_$i" LD_LIBRARY_PATH=$L)
  b=$(run_ppl "ovr1010_$i" HSA_OVERRIDE_GFX_VERSION=10.1.0)
  c=$(run_ppl "ovr1030_$i" HSA_OVERRIDE_GFX_VERSION=10.3.0)
  log "  round $i: native(gfx1013)=$a  override 10.1.0=$b  override 10.3.0=$c"
done

log "=== B. does decode rate track the queue the process is given?"
for i in $(seq 1 10); do
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 900 "$HIP/llama-bench" -m $M8 -lm mmap -ngl 99 -fa 1 -p 0 -n 8 -d 16128 -r 1 \
    > "$D/q_$i.log" 2>&1 &
  bp=$!
  sleep 35
  sudo cat /sys/kernel/debug/kfd/rls > "$D/rls_$i.txt" 2>/dev/null
  sudo cat /sys/kernel/debug/kfd/mqds > "$D/mqds_$i.txt" 2>/dev/null
  wait $bp 2>/dev/null
  rate=$(grep -aoE "tg8 @ d[0-9]+ \| +[0-9.]+" "$D/q_$i.log" | grep -oE "[0-9.]+$")
  db=$(grep -aoiE "doorbell[ _]?id[: ]+0x[0-9a-f]+" "$D/mqds_$i.txt" 2>/dev/null | head -1 | grep -oE "0x[0-9a-f]+")
  qid=$(grep -aoiE "queue id: [0-9]+" "$D/mqds_$i.txt" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  vmid=$(grep -aoiE "vmid: [0-9]+|vmid [0-9]+" "$D/mqds_$i.txt" 2>/dev/null | head -1 | grep -oE "[0-9]+$")
  log "  run $i: rate=${rate:-FAIL} doorbell=${db:-none} queue=${qid:-none} vmid=${vmid:-none}"
done
touch "$D/DONE"; log done
