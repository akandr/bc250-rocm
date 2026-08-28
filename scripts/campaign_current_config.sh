#!/usr/bin/env bash
# Re-run the headline campaign on the configuration the README now recommends.
#
# The five-model throughput table and the four-model correctness gates on the
# front page were measured before three things changed underneath them: the
# navi12 SDMA microcode, kernel 7.1.8, and amdgpu.gpu_recovery=0. The SDMA
# substitution was re-checked against a subset at the time, and gpu_recovery was
# checked against the 1.5B only. The table itself has never been reproduced end
# to end on the current stack, and it is the part of this repository a visitor
# reads first.
#
# Same models, same flags and the same two backends as the original, so the
# numbers are comparable rather than merely fresh.
set -u
exec 9>~/.campaign.lock; flock -n 9 || { echo "another instance holds the lock"; exit 1; }
D=~/inv115; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
VK=~/llama-master/build-vk/bin
WIKI=~/wiki.test.raw
log () { echo "[$(date +%T)] $*" | tee -a "$D/log"; sync; }

log "=== configuration: kernel $(uname -r), gpu_recovery=$(cat /sys/module/amdgpu/parameters/gpu_recovery)"
log "    cmdline: $(tr ' ' '\n' < /proc/cmdline | grep amdgpu | tr '\n' ' ')"

declare -a MODELS=(
  "qwen2.5-1.5b-q4km.gguf|qwen2.5-1.5B Q4_K_M"
  "qwen3-8b-q8_0.gguf|qwen3-8B Q8_0"
  "deepseek-r1-14b.gguf|deepseek-r1-14B Q4_K_M"
  "qwen3-14b.gguf|qwen3-14B Q4_K_M"
  "qwen3.6-35b-a3b-iq2m.gguf|qwen3.6-35B-A3B MoE IQ2_M"
)

log "=== throughput, pp512 and tg64, both backends"
for entry in "${MODELS[@]}"; do
  f=${entry%%|*}; name=${entry#*|}
  tag=$(echo "$f" | sed 's/\.gguf//')
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 30 2400 "$HIP/llama-bench" -m "/opt/models/$f" -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 2 \
    > "$D/hip_$tag.log" 2>&1
  hp=$(grep -aoE "pp512 \| +[0-9.]+" "$D/hip_$tag.log" | grep -oE "[0-9.]+$")
  ht=$(grep -aoE "tg64 \| +[0-9.]+" "$D/hip_$tag.log" | grep -oE "[0-9.]+$")
  timeout -k 30 2400 "$VK/llama-bench" -m "/opt/models/$f" -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 2 \
    > "$D/vk_$tag.log" 2>&1
  vp=$(grep -aoE "pp512 \| +[0-9.]+" "$D/vk_$tag.log" | grep -oE "[0-9.]+$")
  vt=$(grep -aoE "tg64 \| +[0-9.]+" "$D/vk_$tag.log" | grep -oE "[0-9.]+$")
  log "  $name: HIP pp512=${hp:-FAIL} tg64=${ht:-FAIL} | VK pp512=${vp:-FAIL} tg64=${vt:-FAIL}"
done

log "=== correctness gates, wikitext perplexity, context 2048 over eight chunks"
declare -a GATES=(
  "qwen3-8b-q8_0.gguf|qwen3-8B Q8_0"
  "qwen3-14b.gguf|qwen3-14B Q4_K_M"
  "deepseek-r1-14b.gguf|deepseek-r1-14B Q4_K_M"
  "qwen3.6-35b-a3b-iq2m.gguf|qwen3.6-35B-A3B MoE IQ2_M"
)
for entry in "${GATES[@]}"; do
  f=${entry%%|*}; name=${entry#*|}
  tag=$(echo "$f" | sed 's/\.gguf//')
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 60 5400 "$HIP/llama-perplexity" -m "/opt/models/$f" --no-mmap -ngl 99 -fa on \
    -c 2048 -f "$WIKI" --chunks 8 > "$D/gate_hip_$tag.log" 2>&1
  gh=$(grep -aoE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+" "$D/gate_hip_$tag.log" | sed 's/Final estimate: PPL = //')
  timeout -k 60 5400 "$VK/llama-perplexity" -m "/opt/models/$f" --no-mmap -ngl 99 -fa on \
    -c 2048 -f "$WIKI" --chunks 8 > "$D/gate_vk_$tag.log" 2>&1
  gv=$(grep -aoE "Final estimate: PPL = [0-9.]+ \+/- [0-9.]+" "$D/gate_vk_$tag.log" | sed 's/Final estimate: PPL = //')
  log "  $name: HIP ${gh:-FAIL} | VK ${gv:-FAIL}"
done

log "=== faults during the campaign"
log "  this boot: $(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -cE 'page fault \(src_id|GPU reset begin|Queue preemption failed')"
touch "$D/DONE"; log done
