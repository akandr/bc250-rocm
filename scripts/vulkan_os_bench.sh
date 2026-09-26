#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# llama.cpp Vulkan bench on either OS (same kernel 7.1.8). Usage: vkab.sh <label>
export PATH=/usr/bin:/usr/sbin
L=$1; D=~/s0915/vkab; mkdir -p $D; M=/opt/models
if grep -q "Forty Four" /etc/fedora-release; then H=~/llama-master/build-vk-f44/bin; else H=~/llama-master/build-vk/bin; fi
{ cat /etc/fedora-release; echo "H=$H"; vulkaninfo --summary 2>/dev/null | grep -E "driverInfo|deviceName|apiVersion" | head -3; rpm -q mesa-vulkan-drivers; } > $D/$L.env
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0; do
  GGML_VK_VISIBLE_DEVICES=0 timeout -k 20 1500 $H/llama-bench -m $M/$m.gguf -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 5 > $D/${L}_$m.log 2>&1
  echo "$L $m: $(grep -aE "\| +(pp512|tg64) " $D/${L}_$m.log | awk -F"|" "{print \$(NF-2) \$(NF-1)}" | tr -s " " | tr "\n" " ") $(grep -aoE "Vulkan|ROCm" $D/${L}_$m.log | sort -u | tr "\n" " ")"
done
