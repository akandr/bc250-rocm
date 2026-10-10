#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Context ceilings, ROCm against Vulkan, with the KV cache at f16, q8_0 and q4_0.
#
# logs/fedora44-ceilings-2026-09-16/ measured ROCm alone, f16 cache, on the three-patch build, and
# logs/nommap-ceiling-2026-09-22/ compared Vulkan at one point only. This runs both backends from the
# thirteen-patch tree on the same models with the same flags: mmap on (llama-bench's default, as the
# earlier ceilings), -fa on, llama-bench's default batch sizes, one depth per invocation, decode after
# the given depth.
#
# For each model, cache type and backend it climbs a doubling ladder and stops at the first depth
# that fails. A smaller cache cannot fail where a larger one fitted, so q8_0 starts at the depth
# where f16 failed and q4_0 where q8_0 failed; the expensive deep fills run only where they can say
# something. Arms are interleaved at every depth. After any failure a short depth-0 run checks the GPU
# still works, and the run stops on a kernel fault line or an unusable GPU instead of recording what
# follows as ceilings. Free memory and swap are logged with every point, since a run that only
# completes by swapping is not a usable context.
set -u
O=${1:-~/ceilings-compare}; mkdir -p "$O"
L=/opt/bc250-rocm/lib64
HIP=~/llama-master/build-hip-pkf16/bin
VK=~/llama-master/build-vk-f44/bin
TMO=${TMO:-5400}
export HSA_ENABLE_SDMA=0

# model:deepest depth tried
MODELS=${MODELS:-"qwen2.5-1.5b-q4km:131072 qwen3-8b-q8_0:65536 qwen3.6-35b-a3b-iq2m:65536 qwen3-14b:65536 qwen3.8-27b-iq3xxs:32768"}
KVS=${KVS:-"f16 q8_0 q4_0"}

PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$O/log"; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
mem () { free -m | awk '/^Mem:/{a=$7} /^Swap:/{s=$3} END{printf "avail=%sM swap=%sM", a, s}'; }
swo () { awk '/^pswpout/{print $2}' /proc/vmstat; }
bin () { [ "$1" = hip ] && echo "$HIP" || echo "$VK"; }

# one backend model kv depth: prints the tg64 rate, or fails/timeout with the first error line
one () {
  local be=$1 m=$2 kv=$3 d=$4 f t0 t rc v e
  f=$O/${be}_${m}_${kv}_d$d.log
  t0=$(date +%s)
  LD_LIBRARY_PATH=$L timeout -k 30 "$TMO" "$(bin "$be")/llama-bench" -m /opt/models/$m.gguf -ngl 99 \
      -fa on -ctk "$kv" -ctv "$kv" -p 0 -n 64 -d "$d" -r 1 > "$f" 2>&1
  rc=$?
  v=$(grep -aoE "tg64 @ d$d +\| +[0-9.]+" "$f" | grep -oE "[0-9.]+$")
  e=$(grep -aoiE "out of memory|ErrorOutOfDeviceMemory|failed to allocate[^,]*|exceeds resident[^,]*|error[^,]*" "$f" | head -1)
  t=$(( $(date +%s) - t0 ))
  if [ -n "$v" ]; then echo "ok $v ${t}s"
  elif [ "$rc" = 124 ] || { [ "$rc" = 137 ] && [ "$t" -ge "$TMO" ]; }; then echo "timeout - ${t}s"
  elif [ "$rc" = 137 ]; then echo "killed - ${t}s rc=137 SIGKILL before the timeout, see the kernel log for the OOM killer"
  else echo "fails - ${t}s rc=$rc ${e:-no error line}"; fi
}

sane () {  # the GPU still runs a small model at depth 0
  LD_LIBRARY_PATH=$L timeout -k 30 300 "$(bin "$1")/llama-bench" -m /opt/models/qwen2.5-1.5b-q4km.gguf \
      -ngl 99 -fa on -p 0 -n 16 -r 1 2>/dev/null | grep -aqE "tg16 +\| +[0-9.]+"
}

for u in ollama hw-watcher.timer crond; do
  if [ "$(systemctl is-active $u 2>/dev/null)" = active ]; then sudo systemctl stop $u; echo $u >> "$O/stopped"; fi
done
restore () { [ -f "$O/stopped" ] && while read -r u; do sudo systemctl start "$u"; done < "$O/stopped"; }
trap restore EXIT
sleep 5

B=$(faults); THR0=$(sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat)
log "start: faults already in this boot $B, edge $(temp), $(mem), stopped: $(tr '\n' ' ' < "$O/stopped" 2>/dev/null)"
log "oberon max: $(awk '/max:/{print $3; exit}' /etc/oberon-config.yaml), clock now: $(grep '\*' /sys/class/drm/card*/device/pp_dpm_sclk | tr -s ' ')"
log "kfd: $(sudo cat /sys/kernel/debug/kfd/mem_limit 2>/dev/null | tr '\n' ' ')"
printf "backend\tmodel\tkv\tdepth\tresult\trate\tseconds\tswapped_out_pages\n" > "$O/summary.tsv"

declare -A START   # first depth to try, per backend and model, raised as the cache shrinks
declare -A alive from
for kv in $KVS; do
  for entry in $MODELS; do
    m=${entry%%:*}; top=${entry#*:}
    for be in hip vk; do
      alive[$be]=1
      s=${START[$be.$m]:-8192}
      [ "$s" -gt "$top" ] && { log "$be $m $kv: skipped, fitted at d$top with a larger cache"; alive[$be]=0; }
      from[$be]=$s
    done
    d=8192
    while [ "$d" -le "$top" ]; do
      for be in hip vk; do
        [ "${alive[$be]}" = 1 ] || continue
        [ "$d" -lt "${from[$be]}" ] && continue
        s0=$(swo)
        r=$(one "$be" "$m" "$kv" "$d")
        sw=$(( $(swo) - s0 ))
        log "$be $m $kv d$d: $r edge=$(temp) $(mem) swapped-out-pages=$sw"
        set -- $r
        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n" "$be" "$m" "$kv" "$d" "$1" "$2" "$3" "$sw" >> "$O/summary.tsv"
        if [ "$1" != ok ]; then
          alive[$be]=0; START[$be.$m]=$d
          [ "$be" = hip ] && log "  kfd after failure: $(sudo cat /sys/kernel/debug/kfd/mem_limit 2>/dev/null | tr '\n' ' ')"
          sane "$be" || { log "GPU NOT USABLE after $be $m $kv d$d, stopping"; exit 1; }
        fi
        n=$(faults); [ "$n" -gt "$B" ] && { log "KERNEL FAULT after $be $m $kv d$d, stopping"; exit 1; }
        sleep 20
      done
      [ "${alive[hip]}" = 0 ] && [ "${alive[vk]}" = 0 ] && break
      d=$(( d * 2 ))
    done
    for be in hip vk; do [ "${alive[$be]}" = 1 ] && START[$be.$m]=$(( top * 2 )); done
  done
done
t=$(sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat)
log "governor overheat lines during the run: $(( t - THR0 ))"
log DONE
