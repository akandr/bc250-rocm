#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Same llama-bench on either OS; records OS, power state and results. Usage: osab.sh <label>
export PATH=/usr/bin:/usr/sbin HSA_ENABLE_SDMA=0
L=$1; D=~/s0915/osab; mkdir -p $D; M=/opt/models
if grep -q "Forty Four" /etc/fedora-release; then
  H=~/llama-master/build-hip-f44/bin; export LD_LIBRARY_PATH=$HOME/rb711/comgr-fixed:$HOME/rb711/install/lib64
else
  H=~/llama-master/build-hip/bin; export LD_LIBRARY_PATH=$HOME/rocblas-f16patch/lib
fi
C=$(ls -d /sys/class/drm/card*/device | head -1)
{ echo "label=$L os=$(cat /etc/fedora-release) boot=$(cat /proc/sys/kernel/random/boot_id) uptime=$(cut -d" " -f1 /proc/uptime)"
  echo "H=$H LD_LIBRARY_PATH=$LD_LIBRARY_PATH"; echo "tuned=$(tuned-adm active 2>/dev/null) oberon=$(systemctl is-active oberon-governor)"
  echo "dpm=$(cat $C/power_dpm_force_performance_level) cmdline=$(cat /proc/cmdline)"; sensors 2>/dev/null | grep -iE "edge|Tctl" | head -2; } > $D/$L.env
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0; do
  timeout -k 20 1500 $H/llama-bench -m $M/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 5 > $D/${L}_$m.log 2>&1
  echo "$L $m: $(grep -aE "pp512|tg64" $D/${L}_$m.log | awk -F"|" "{print \$(NF-1)}" | tr -s " " | tr "\n" " ")"
done
