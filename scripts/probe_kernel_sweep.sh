#!/bin/bash
# sync-per-line so data survives a hard wedge + power-cycle
LOG=~/kerneltest715.log
{ echo "===== FRESH SAMPLE kernel=$(uname -r) CU=$(journalctl -k -b 0|grep -o "active_cu_number [0-9]*"|tail -1) up=$(cut -d. -f1 /proc/uptime)s $(date +%T) ====="; } >> $LOG; sync
for nb in 4096 32768; do
  echo "--- nblocks=$nb ($((nb*256)) threads) ---" >> $LOG; sync
  HSA_ENABLE_SDMA=0 ~/bin/timeout 90 ~/compute_probe $nb 2>&1 | grep --line-buffered -iE "iter|RESULT|MISMATCH|wrong|HSA_STATUS|fault|abort" | while IFS= read -r l; do echo "$l" >> $LOG; sync; done
  echo "  [done nb=$nb]" >> $LOG; sync
  sleep 3
done
echo "===== SAMPLE END $(date +%T) =====" >> $LOG; sync
