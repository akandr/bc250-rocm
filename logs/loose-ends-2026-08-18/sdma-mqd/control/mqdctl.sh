#!/usr/bin/env bash
# Control: do queue descriptors change while work is actually running?
# If the compute queue descriptor moves during compute work, descriptors track
# live state and the static SDMA descriptor during a hang is meaningful. If it
# does not, the MQD is only a save area and the SDMA observation proves nothing.
set -u
export HSA_ENABLE_SDMA=0
~/compute_probe 32768 6000 3 > /tmp/cp_ctl.log 2>&1 &
P=$!
sleep 3
for i in 1 2 3 4; do sudo cat /sys/kernel/debug/kfd/mqds > /tmp/mqd_c$i.txt 2>/dev/null; sleep 3; done
wait $P 2>/dev/null
echo "compute probe: $(grep -aoE "ALL CORRECT|total_wrong=[0-9]+" /tmp/cp_ctl.log | tail -1)"
for i in 2 3 4; do
  if diff -q /tmp/mqd_c1.txt /tmp/mqd_c$i.txt >/dev/null 2>&1; then echo "  compute sample $i: identical to sample 1"
  else echo "  compute sample $i: CHANGED ($(diff /tmp/mqd_c1.txt /tmp/mqd_c$i.txt | grep -c "^[<>]") differing lines)"; fi
done
