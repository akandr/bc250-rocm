#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Where does Fedora 44 gain speed? Same probes on either OS. Usage: osprobe.sh <label>
export PATH=/usr/bin:/usr/sbin HSA_ENABLE_SDMA=0
L=$1; D=~/s0915/osprobe; mkdir -p $D; M=/opt/models/qwen2.5-1.5b-q4km.gguf
if grep -q "Forty Four" /etc/fedora-release; then
  H=~/llama-master/build-hip-f44/bin; export LD_LIBRARY_PATH=$HOME/rb711/comgr-fixed:$HOME/rb711/install/lib64
else
  H=~/llama-master/build-hip/bin; export LD_LIBRARY_PATH=$HOME/rocblas-f16patch/lib
fi
C=$(ls -d /sys/class/drm/card*/device | head -1)
b() { tag=$1; shift; env "$@" timeout -k 20 900 $H/llama-bench -m $M -mmp 0 -ngl 99 -r 3 "${BARGS[@]}" > $D/${L}_$tag.log 2>&1
  echo "$L $tag: $(grep -aE "\| +(pp|tg)[0-9]+ " $D/${L}_$tag.log | awk -F"|" "{print \$(NF-2) \$(NF-1)}" | tr -s " " | tr "\n" " ")"; }
BARGS=(-fa on -p 512 -n 64); b default X=1
BARGS=(-fa on -p 512 -n 64); b nographs GGML_CUDA_DISABLE_GRAPHS=1
BARGS=(-fa off -p 512 -n 64); b faoff X=1
BARGS=(-fa on -p 2048 -ub 2048 -b 2048 -n 0); b pp2048_ub2048 X=1
BARGS=(-fa on -p 0 -n 256); 
true
/usr/bin/time -f "%e wall %U user %S sys" -o $D/${L}_tg256.time env timeout -k 20 900 $H/llama-bench -m $M -mmp 0 -ngl 99 -r 1 -fa on -p 0 -n 256 > $D/${L}_tg256.log 2>&1
wait
echo "$L tg256: $(grep -aE "tg256" $D/${L}_tg256.log | awk -F"|" "{print \$(NF-1)}") time: $(cat $D/${L}_tg256.time) "
