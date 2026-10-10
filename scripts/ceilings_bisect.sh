#!/usr/bin/env bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Context ceilings found to the real limit: ROCm with HIP graphs, ROCm without them, and Vulkan, with
# the KV cache at f16, q8_0 and q4_0.
#
# scripts/ceilings_compare.sh (logs/context-ceilings-2026-09-30/earlier-passes/) climbed a doubling ladder with a
# depth cap per model, so several cells ended "still generating at the deepest depth tried", and every
# ceiling it did find is only known to lie somewhere inside a doubling. This finds each one: it climbs
# past the deepest pass until a depth fails, with no cap, then bisects between the last pass and the
# first failure down to an eighth of that interval (RESDIV sets the fraction).
#
# Four arms. hip is ROCm as shipped. hipng is the same build with GGML_CUDA_DISABLE_GRAPHS=1, and hippc
# keeps HIP graphs but sets DEBUG_CLR_GRAPH_PACKET_CAPTURE=0: instantiating a graph with clr's AQL packet
# capture on is what failed for memory at the first generated token (this harness's first pass, in
# logs/context-ceilings-2026-09-30/earlier-passes/). vk
# is Vulkan.
#
# llama-bench as shipped stores the sequence state in a host buffer after the depth fill
# (llama_state_seq_get_data, also with -r 1, where it is never reused). On ROCm clr pins that buffer, and
# the pinned pages count against the KFD limit until the process exits; on Vulkan it costs the same in
# RAM. That copy, as large as the KV cache, is what left no memory for the packet capture above, which
# itself needs about 20 MB. HIP and VK name the directories holding the llama-bench to run: pointed at one
# built with llama-bench-no-depth-cache.patch and with LLAMA_BENCH_NO_DEPTH_CACHE=1 exported, this
# measures the ceilings without the copy (scripts/ceilings_nosnap_chain.sh).
#
# Known results seed the search: every summary.tsv named in SEED counts, and so does this run's own, so an
# interrupted campaign resumes where it stopped. Memory decides these ceilings, so a smaller cache
# cannot fail where a larger one fitted, a larger one cannot fit where a smaller one failed, and nothing
# fails below a depth that passed; the search uses that instead of re-running what it implies. Arms are
# kept apart: nothing measured on one is assumed for another.
#
# Every run samples KFD's memory accounting, GTT use and free memory every five seconds (runs/*.mem).
# After a failure a depth-0 run checks the GPU still works, and the campaign stops on a kernel fault line
# or an unusable GPU instead of recording what follows as ceilings.
set -u
O=${1:-~/ceilings-bisect}; mkdir -p "$O/runs"
L=/opt/bc250-rocm/lib64
HIP=${HIP:-~/llama-master/build-hip-pkf16/bin}
VK=${VK:-~/llama-master/build-vk-f44/bin}
TMO=${TMO:-86400}          # a day per run: a slow fill is not a failure
MAXD=${MAXD:-4194304}
RESDIV=${RESDIV:-8}        # bisect until the bracket is 1/RESDIV of the one it started from
export HSA_ENABLE_SDMA=0
MODELS=${MODELS:-"qwen3-14b qwen3-8b-q8_0 qwen3.8-27b-iq3xxs qwen3.6-35b-a3b-iq2m qwen2.5-1.5b-q4km"}
KVS=${KVS:-"f16 q8_0 q4_0"}
ARMS=${ARMS:-"hip hipng hippc vk"}
SEED=${SEED:-}
SUM=$O/summary.tsv

PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
faults () { sudo journalctl -b 0 -k --no-pager 2>/dev/null | grep -ciE "$PAT"; }
log () { echo "[$(date '+%F %T')] $*" | tee -a "$O/log" >&2; sync; }
temp () { sensors 2>/dev/null | awk '/edge/{print $2}'; }
mem () { free -m | awk '/^Mem:/{a=$7} /^Swap:/{s=$3} END{printf "avail=%sM swap=%sM", a, s}'; }
swo () { awk '/^pswpout/{print $2}' /proc/vmstat; }
bin () { [ "$1" = vk ] && echo "$VK" || echo "$HIP"; }
envof () {
  case $1 in
    hipng) echo "GGML_CUDA_DISABLE_GRAPHS=1" ;;
    hippc) echo "DEBUG_CLR_GRAPH_PACKET_CAPTURE=0" ;;
    *) echo "X=1" ;;
  esac
}

# ---- known results: R[arm|model|kv|depth] = ok|fail ----
declare -A R
declare -A SIZE=( [f16]=3 [q8_0]=2 [q4_0]=1 )
load () {  # a summary.tsv from either harness: first column is the backend or arm
  [ -f "$1" ] || return 0
  while IFS=$'\t' read -r a m kv d res rate secs rest; do
    [ "$a" = backend ] || [ "$a" = arm ] && continue
    case $res in
      ok) R["$a|$m|$kv|$d"]=ok ;;
      fails|killed) R["$a|$m|$kv|$d"]=fail ;;
      # the first harness wrote OOM kills as timeout; a real timeout is not a result
      timeout) [ "${secs%s}" -lt 3600 ] && R["$a|$m|$kv|$d"]=fail ;;
    esac
  done < "$1"
}
for f in $SEED "$SUM"; do load "$f"; done
[ -f "$SUM" ] || printf "arm\tmodel\tkv\tdepth\tresult\trate\tseconds\tswapped_out_pages\n" > "$SUM"

# lo: deepest depth known to pass, for this cache or a larger one; hi: shallowest known to fail, for
# this cache or a smaller one. Empty when unknown.
bounds () {
  local a=$1 m=$2 kv=$3 k v lo="" hi="" ka km kk kd
  for k in "${!R[@]}"; do
    IFS='|' read -r ka km kk kd <<< "$k"
    [ "$ka" = "$a" ] && [ "$km" = "$m" ] || continue
    v=${R[$k]}
    if [ "$v" = ok ] && [ "${SIZE[$kk]}" -ge "${SIZE[$kv]}" ]; then
      [ -z "$lo" ] || [ "$kd" -gt "$lo" ] && lo=$kd
    elif [ "$v" = fail ] && [ "${SIZE[$kk]}" -le "${SIZE[$kv]}" ]; then
      [ -z "$hi" ] || [ "$kd" -lt "$hi" ] && hi=$kd
    fi
  done
  echo "${lo:-0} ${hi:-0}"
}

sane () {
  env $(envof "$1") LD_LIBRARY_PATH=$L timeout -k 30 300 "$(bin "$1")/llama-bench" \
      -m /opt/models/qwen2.5-1.5b-q4km.gguf -ngl 99 -fa on -p 0 -n 16 -r 1 2>/dev/null |
      grep -aqE "tg16 +\| +[0-9.]+"
}

# run arm model kv depth: records the result and sets RES to ok|fail|timeout. Not called in a subshell,
# so the result it adds to R is seen by the search.
run () {
  local a=$1 m=$2 kv=$3 d=$4 f t0 t rc v e sp s0 sw res rate
  f=$O/runs/${a}_${m}_${kv}_d$d
  ( while :; do
      echo "$(date +%s) | $(sudo cat /sys/kernel/debug/kfd/mem_limit 2>/dev/null | tr '\n' ' ') |" \
           "gtt=$(cat /sys/class/drm/card*/device/mem_info_gtt_used 2>/dev/null | head -1)" \
           "avail=$(free -m | awk '/^Mem:/{print $7}')M"
      sleep 5
    done ) > "$f.mem" &
  sp=$!
  s0=$(swo); t0=$(date +%s)
  env $(envof "$a") LD_LIBRARY_PATH=$L timeout -k 30 "$TMO" "$(bin "$a")/llama-bench" -m /opt/models/$m.gguf \
      -ngl 99 -fa on -ctk "$kv" -ctv "$kv" -p 0 -n 64 -d "$d" -r 1 > "$f.log" 2>&1
  rc=$?
  t=$(( $(date +%s) - t0 )); sw=$(( $(swo) - s0 ))
  kill "$sp" 2>/dev/null; wait "$sp" 2>/dev/null
  v=$(grep -aoE "tg64 @ d$d +\| +[0-9.]+" "$f.log" | grep -oE "[0-9.]+$")
  e=$(grep -aoiE "ROCm error|ErrorOutOfDeviceMemory|Failed to allocate[^,]*|out of memory" "$f.log" | head -1)
  if [ -n "$v" ]; then res=ok; rate=$v
  elif [ "$rc" = 124 ] || { [ "$rc" = 137 ] && [ "$t" -ge "$TMO" ]; }; then res=timeout; rate=-
  elif [ "$rc" = 137 ]; then res=killed; rate=-; e="SIGKILL before the timeout (OOM killer, see the kernel log)"
  else res=fails; rate=-; fi
  printf "%s\t%s\t%s\t%s\t%s\t%s\t%ss\t%s\n" "$a" "$m" "$kv" "$d" "$res" "$rate" "$t" "$sw" >> "$SUM"
  log "$a $m $kv d$d: $res $rate ${t}s rc=$rc ${e:-} edge=$(temp) $(mem) swapped-out-pages=$sw"
  case $res in
    ok) R["$a|$m|$kv|$d"]=ok; RES=ok ;;
    timeout) RES=timeout ;;
    *) R["$a|$m|$kv|$d"]=fail
       sane "$a" || { log "GPU NOT USABLE after $a $m $kv d$d, stopping"; exit 1; }
       RES=fail ;;
  esac
}

for u in ollama hw-watcher.timer crond; do
  if [ "$(systemctl is-active $u 2>/dev/null)" = active ]; then sudo systemctl stop $u; echo $u >> "$O/stopped"; fi
done
restore () { [ -f "$O/stopped" ] && while read -r u; do sudo systemctl start "$u"; done < "$O/stopped"; rm -f "$O/stopped"; }
trap restore EXIT
sleep 5

F0=$(faults); THR0=$(sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat)
log "start: seeds '$SEED', ${#R[@]} known results, faults in this boot $F0, edge $(temp), $(mem)"
log "oberon max: $(awk '/max:/{print $3; exit}' /etc/oberon-config.yaml); kfd: $(sudo cat /sys/kernel/debug/kfd/mem_limit | tr '\n' ' ')"

step () {  # one run with the checks around it; returns 1 when the cell has to be abandoned
  run "$@"
  [ "$(faults)" -gt "$F0" ] && { log "KERNEL FAULT after $*, stopping"; exit 1; }
  sleep 20
  [ "$RES" = timeout ] && { log "  $*: timed out after ${TMO}s, cell abandoned"; return 1; }
  return 0
}

for m in $MODELS; do
  for kv in $KVS; do
    for a in $ARMS; do
      read -r lo hi <<< "$(bounds "$a" "$m" "$kv")"
      log "cell $a $m $kv: known pass <= $lo, known fail >= $hi"
      ok=1
      # no pass known: start at 8192 and halve until something passes
      if [ "$lo" = 0 ]; then
        d=8192
        while [ "$d" -ge 1024 ] && [ "$ok" = 1 ]; do
          step "$a" "$m" "$kv" "$d" || { ok=0; break; }
          read -r lo hi <<< "$(bounds "$a" "$m" "$kv")"
          [ "$lo" != 0 ] && break
          d=$(( d / 2 ))
        done
        [ "$lo" = 0 ] && { log "  $a $m $kv: fails even at 1024"; continue; }
      fi
      # no failure known: double until one fails
      while [ "$ok" = 1 ] && [ "$hi" = 0 ]; do
        d=$(( lo * 2 ))
        [ "$d" -gt "$MAXD" ] && { log "  $a $m $kv: still passing at $lo, MAXD reached"; ok=0; break; }
        step "$a" "$m" "$kv" "$d" || { ok=0; break; }
        read -r lo hi <<< "$(bounds "$a" "$m" "$kv")"
      done
      [ "$ok" = 1 ] || continue
      # bisect down to 1/RESDIV of the bracket (an eighth by default), in steps of 1024 tokens
      res=$(( (hi - lo) / RESDIV )); [ "$res" -lt 1024 ] && res=1024
      while [ $(( hi - lo )) -gt "$res" ]; do
        d=$(( (lo + hi) / 2 / 1024 * 1024 ))
        [ "$d" -le "$lo" ] && d=$(( lo + 1024 ))
        [ "$d" -ge "$hi" ] && break
        step "$a" "$m" "$kv" "$d" || { ok=0; break; }
        read -r lo hi <<< "$(bounds "$a" "$m" "$kv")"
      done
      [ "$ok" = 1 ] && log "  RESULT $a $m $kv: generates at $lo, fails at $hi"
    done
  done
done
t=$(sudo journalctl -b 0 -u oberon-governor --no-pager 2>/dev/null | grep -ci overheat)
log "governor overheat lines during the run: $(( t - THR0 ))"
log DONE
