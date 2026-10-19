#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-only
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Which endings of a GPU process leave system memory behind on this board? Three times in two days
# MemAvailable fell from about 14.4 GiB to 1 to 4 GiB with no process holding it, once right after a ROCm
# run aborted at load (ggml-cuda.cu:111, core dump), twice overnight while only ollama's Vulkan runner had
# used the GPU. This script provokes each ending in turn on the 1.5B and reads MemAvailable, KFD's own
# accounting and the GPU's GTT counter before and after:
#   1. a ROCm llama-bench that exits normally
#   2. a Vulkan llama-bench that exits normally
#   3. a ROCm llama-bench killed with SIGKILL while running
#   4. a Vulkan llama-bench killed with SIGKILL while running
#   5. a ROCm run that aborts at load (the 8B behind 40960 tokens does not fit; stock llama-bench)
#   6. ollama loading a model and unloading it the way its scheduler does
# Each step waits ten seconds and logs the three counters; a drop that stays is a leak. Run on a freshly
# rebooted board, and reboot afterwards, since the leaks found do not clear.
set -u
O=${1:-$HOME/leak-test-2026-10-18}; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=$HOME/llama-master/build-hip-pkf16/bin
VK=$HOME/llama-master/build-vk-f44/bin
M15=/opt/models/qwen2.5-1.5b-q4km.gguf
M8=/opt/models/qwen3-8b-q8_0.gguf
export HSA_ENABLE_SDMA=0
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
counters () {
  local a k g
  a=$(free -m | awk '/^Mem:/{print $7}')
  k=$(sudo cat /sys/kernel/debug/kfd/mem_limit 2>/dev/null | tr '\n' ' ')
  g=$(cat /sys/class/drm/card0/device/mem_info_gtt_used 2>/dev/null)
  echo "avail ${a} MiB; kfd: ${k}; gtt_used $((g / 1048576)) MiB"
}
STOPPED=""
for u in ollama hw-watcher.timer crond dnf-makecache.timer sysstat-collect.timer plocate-updatedb.timer; do
  if [ "$(systemctl is-active $u 2>/dev/null)" = active ]; then sudo systemctl stop $u; STOPPED="$STOPPED $u"; fi
done
restore () { for u in $STOPPED; do sudo systemctl start $u; done; log "restored:$STOPPED"; }
trap restore EXIT
sleep 5
log "0 baseline: $(counters)"

step () {  # $1 label, $2 bin dir, $3 model, $4 mode (exit|kill|abort), rest: extra llama-bench args
  local tag=$1; local b=$2; local m=$3; local mode=$4; shift 4
  log "$tag before: $(counters)"
  LD_LIBRARY_PATH=$L timeout -k 30 900 "$b/llama-bench" -m "$m" -mmp 0 -ngl 99 -fa on "$@" > "$O/raw_$tag" 2> "$O/err_$tag" &
  local pid=$!
  if [ "$mode" = kill ]; then
    # wait until the model is loaded and work has started (GTT climbs past 500 MiB and the process is still alive)
    for i in $(seq 1 120); do sleep 1; g=$(cat /sys/class/drm/card0/device/mem_info_gtt_used); [ $((g / 1048576)) -gt 500 ] && break; done
    sleep 5
    log "$tag loaded, $(counters); killing with SIGKILL"
    sudo kill -9 $(pgrep -x llama-bench) 2>/dev/null
  fi
  wait $pid; local rc=$?
  sleep 10
  log "$tag after (rc $rc): $(counters)"
  grep -a -E "tg64|pp512|error|abort" "$O/raw_$tag" "$O/err_$tag" 2>/dev/null | grep -v -i "terminal\|load-mode" | head -2 | cut -c1-160 | sed 's/^/    /' | tee -a "$O/log"
}
step 1_hip_exit  "$HIP" "$M15" exit  -p 0 -n 64 -r 1
step 2_vk_exit   "$VK"  "$M15" exit  -p 0 -n 64 -r 1
step 3_hip_kill  "$HIP" "$M15" kill  -p 512 -n 512 -r 5
step 4_vk_kill   "$VK"  "$M15" kill  -p 512 -n 512 -r 5
step 5_hip_abort "$HIP" "$M8"  abort -p 0 -n 64 -d 40960 -r 1

# 6: ollama's own load and unload
sudo systemctl start ollama; sleep 8
log "6 ollama before: $(counters)"
MODEL=$(ollama list 2>/dev/null | awk 'NR==2{print $1}')
log "6 ollama model: $MODEL"
if [ -n "$MODEL" ]; then
  curl -s -o /dev/null http://127.0.0.1:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"Say hi.\",\"stream\":false,\"keep_alive\":\"30s\"}"
  log "6 ollama loaded: $(counters)"
  sleep 60
  log "6 ollama 60 s later (keep_alive 30 s, runner unloaded by its scheduler): $(counters); runners: $(pgrep -f 'ollama runne[r]' | wc -l)"
  curl -s -o /dev/null http://127.0.0.1:11434/api/generate -d "{\"model\":\"$MODEL\",\"prompt\":\"Say hi.\",\"stream\":false,\"keep_alive\":0}"
  sleep 15
  log "6 ollama after keep_alive 0 (immediate unload): $(counters); runners: $(pgrep -f 'ollama runne[r]' | wc -l)"
fi
sudo systemctl stop ollama; STOPPED="$STOPPED"
log "end: $(counters)"
touch "$O/DONE"; log DONE
