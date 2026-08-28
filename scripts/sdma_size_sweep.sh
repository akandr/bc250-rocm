#!/usr/bin/env bash
# Where does working SDMA actually help, if anywhere?
#
# GabriWar expected the navi12 microcode to help "vulkan and many small tensor
# operations". Vulkan inference showed no change. This tests the other half by
# sweeping transfer size with SDMA on against off, which also documents where
# each path wins rather than asserting it.
#
# A prediction worth stating before measuring: ROCclr stages copies below 16384
# bytes through a blit compute kernel and only hands larger ones to SDMA, so
# below that threshold the two arms should be indistinguishable and the firmware
# irrelevant. If small transfers do improve, that prediction is wrong and the
# path selection is not what the tracing suggested.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv79; mkdir -p "$D"
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

rate () { # rate <bytes> <sdma> -> copies per second
  local out
  out=$(HSA_ENABLE_SDMA=$2 timeout -k 5 30 ~/sdma_one "$1" 6 2>&1 | grep -aoE "COMPLETED copies=[0-9]+" | grep -oE "[0-9]+$")
  echo "${out:-0}"
}

log "=== copies completed in 6 s, SDMA on against off, alternated per size"
printf "%-12s %14s %14s %10s\n" size sdma_on sdma_off ratio | tee -a "$D/log"
for sz in 4096 16384 16385 65536 262144 1048576 16777216; do
  a=$(rate $sz 1); b=$(rate $sz 0); c=$(rate $sz 0); d=$(rate $sz 1)
  on=$(( (a + d) / 2 )); off=$(( (b + c) / 2 ))
  r=$(echo "scale=2; $on / ($off + 0.001)" | bc 2>/dev/null || echo "?")
  printf "%-12s %14s %14s %10s\n" "$sz" "$on" "$off" "$r" | tee -a "$D/log"
done
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out\")"
touch "$D/DONE"; log done
