#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# hws_fullcheck_orchestrate.sh: run hws_fullcheck_boot.sh once per configuration, one boot each.
# Runs on the controlling machine; sets the kernel's boot arguments with grubby, makes it the default,
# reboots, waits for a fresh boot, runs the battery and copies its directory back. The boot loader
# entries and the default are backed up first and restored exactly at the end. Power cycling on a
# hung boot is left to the operator.
# Usage: hws_fullcheck_orchestrate.sh <local out dir> [config file]; a config file holds one
# "tag|kernel|module arguments|HSA_ENABLE_SDMA value or empty" line per boot, run in order.
# BATTERY and RDIR select another per-boot battery and its result directory on the board.
set -u
OUT=${1:?local output dir}; mkdir -p "$OUT"
BATTERY=${BATTERY:-hws_fullcheck_boot.sh}; RDIR=${RDIR:-hwsfull}
BASE="ro rootflags=subvol=root-f44 ttm.pages_limit=4194304 mitigations=off amdgpu.gpu_recovery=0 rhgb quiet"
# tag | kernel | module arguments (hardware scheduling: no sched_policy)
CONFIGS=(
  "7.1.5-40cu-hws-rl3|7.1.5-100.fc43.x86_64|amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3 amdgpu.bc250_cc_write_mode=3"
  "7.1.5-24cu-hws-rl3|7.1.5-100.fc43.x86_64|amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3 amdgpu.bc250_cc_write_mode=0"
  "7.1.5-40cu-hws-bypassonly|7.1.5-100.fc43.x86_64|amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=0 amdgpu.bc250_cc_write_mode=3"
  "7.1.8-40cu-hws-rl3|7.1.8-100.fc43.x86_64|amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=3 amdgpu.bc250_cc_write_mode=3"
  "7.1.2-40cu-hws-rl1|7.1.2-200.fc43.x86_64|amdgpu.bc250_flush_pasid_kiq=0 amdgpu.bc250_flush_by_runlist=1 amdgpu.bc250_cc_write_mode=3"
)
if [ -n "${2:-}" ]; then CONFIGS=(); while IFS= read -r l; do CONFIGS+=("$l"); done < <(grep -v '^#' "$2" | grep .); fi
ssh bc250 'sudo -n rm -rf /root/hwsfull_entries_backup && sudo -n cp -a /boot/loader/entries /root/hwsfull_entries_backup &&
           sudo -n grub2-editenv list | sudo -n tee /root/hwsfull_grubenv_backup >/dev/null && sudo -n grubby --info=ALL > ~/hwsfull_grubby_before.txt'
up() { ssh -o ConnectTimeout=8 -o BatchMode=yes bc250 'cut -d. -f1 /proc/uptime' 2>/dev/null; }
wait_fresh() { local w=0; while [ $w -lt 1200 ]; do u=$(up); [ -n "${u:-}" ] && [ "$u" -lt 400 ] && return 0; sleep 20; w=$((w+20)); done; return 1; }
reboot_into() { # kernel args
  ssh bc250 "sudo -n grubby --update-kernel=/boot/vmlinuz-$1 --args='$BASE $2' --remove-args='amdgpu.sched_policy amdgpu.bc250_flush_by_runlist amdgpu.bc250_flush_pasid_kiq amdgpu.bc250_cc_write_mode' >/dev/null &&
             sudo -n grubby --update-kernel=/boot/vmlinuz-$1 --args='$2' >/dev/null &&
             sudo -n grubby --set-default=/boot/vmlinuz-$1 >/dev/null && sync"
  ssh bc250 'nohup sh -c "sleep 3; sudo -n /sbin/reboot" >/dev/null 2>&1 </dev/null & exit'
  sleep 45; wait_fresh
}
for c in "${CONFIGS[@]}"; do
  IFS='|' read -r TAG K ARGS SDMA <<< "$c"
  echo "=== $TAG ($(date +%H:%M:%S))"
  if ! reboot_into "$K" "$ARGS"; then echo "  BOOT DID NOT COME BACK: power-cycle by hand, then rerun"; exit 1; fi
  ssh bc250 'sudo -n systemctl stop ollama nginx tor signal-cli 2>/dev/null; sleep 20'
  ssh -o ServerAliveInterval=20 -o ServerAliveCountMax=30 bc250 "${SDMA:+HSA_ENABLE_SDMA=$SDMA} nohup setsid ~/fullcheck/$BATTERY $TAG > ~/${RDIR}_$TAG.nohup 2>&1 < /dev/null &"
  w=0; until ssh -o ConnectTimeout=8 bc250 "test -f ~/$RDIR/$TAG/DONE" 2>/dev/null; do
    sleep 30; w=$((w+30)); [ $w -gt 3600 ] && { echo "  battery not DONE after 60 min (wedge?)"; break; }; done
  scp -q -r "bc250:$RDIR/$TAG" "$OUT/" 2>/dev/null
  ssh bc250 "tail -3 ~/$RDIR/$TAG/battery.log" 2>/dev/null | sed 's/^/  /'
done
echo "=== restoring the boot entries and the default"
ssh bc250 'sudo -n rm -rf /boot/loader/entries && sudo -n cp -a /root/hwsfull_entries_backup /boot/loader/entries &&
           sudo -n grub2-editenv - set "$(sudo -n grep ^saved_entry= /root/hwsfull_grubenv_backup)" && sync &&
           sudo -n grubby --info=ALL | diff - ~/hwsfull_grubby_before.txt && echo entries-identical'
ssh bc250 'nohup sh -c "sleep 3; sudo -n /sbin/reboot" >/dev/null 2>&1 </dev/null & exit'
sleep 45; wait_fresh
ssh bc250 'uname -r; cat /proc/cmdline'
echo ALL-DONE
