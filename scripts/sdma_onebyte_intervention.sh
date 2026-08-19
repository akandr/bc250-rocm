#!/usr/bin/env bash
# The one-byte SDMA intervention, second attempt.
#
# The first attempt passed 16384 and 16385 to a probe whose argument is a
# repetition count, not a size, so both arms ran identical 16 KiB copies and the
# experiment compared nothing. It also sampled the queue descriptor after a
# single small copy had already finished, which shows an idle queue whatever the
# truth is. Both are fixed here with a purpose-built probe that takes the size in
# bytes and loops, keeping the queue busy throughout the sampling window.
#
# The claim under test: the SDMA failure sits at the submission end, because the
# queue descriptor never advances during a hang. That rested on comparing a hang
# against a different workload. Here the only difference is one byte.
set -u
D=~/inv74; mkdir -p "$D"
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

log "waiting for the queue to drain"
while [ ! -f ~/inv73/DONE ]; do sleep 60; done

arm () { # arm <tag> <bytes> <expectation>
  local tag=$1 bytes=$2 expect=$3
  log "=== $tag: $bytes bytes ($expect)"
  HSA_ENABLE_SDMA=1 timeout -k 10 45 ~/sdma_one "$bytes" 25 > "$D/$tag.probe" 2>&1 &
  local P=$!
  sleep 5
  for i in 1 2 3 4 5; do
    sudo cat /sys/kernel/debug/kfd/mqds > "$D/${tag}_mqd_$i.txt" 2>/dev/null
    sleep 3
  done
  wait $P 2>/dev/null
  local ch=0
  for i in 2 3 4 5; do
    ch=$(( ch + $(diff "$D/${tag}_mqd_1.txt" "$D/${tag}_mqd_$i.txt" 2>/dev/null | grep -c "^[<>]") ))
  done
  log "  probe: $(grep -aoE "COMPLETED copies=[0-9]+|copies=[0-9]+|failed.*" "$D/$tag.probe" | tail -1 || echo NO-OUTPUT)"
  log "  descriptor lines changed across four samples: $ch"
}

# Alternated so drift cannot masquerade as the effect.
arm below_1 16384 "expected to loop freely"
arm above_1 16385 "expected to hang on the first copy"
arm above_2 16385 "expected to hang on the first copy"
arm below_2 16384 "expected to loop freely"

log "=== control: the hanging size with SDMA disabled, which takes the blit path"
HSA_ENABLE_SDMA=0 timeout -k 10 45 ~/sdma_one 16385 10 > "$D/sdmaoff.probe" 2>&1
log "  probe: $(grep -aoE "COMPLETED copies=[0-9]+|failed.*" "$D/sdmaoff.probe" | tail -1 || echo NO-OUTPUT)"
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
