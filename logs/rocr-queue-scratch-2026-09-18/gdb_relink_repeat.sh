#!/bin/bash
# SPDX-License-Identifier: AGPL-3.0-or-later
# Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
# Assisted-by: Claude (Anthropic)
# Both HIP-side faults were the first run after libhsa-runtime64 had been replaced. Six gdb runs of the
# deep-context command with the installed runtime: before the odd runs the library file is re-copied
# (new inode, new mtime) and its pages and libamdhip64's dropped from the page cache, the even runs are left alone. Waits for gdn-chain.
L=/opt/bc250-rocm/lib64; M=~/llama-master; O=~/rocr-repro
until grep -qE "CHAIN DONE|CHAIN FAILED" ~/gdn-chain/log 2>/dev/null; do sleep 60; done
sleep 60
: > $O/relink.log
for i in 1 2 3 4 5 6; do
  if [ $((i % 2)) -eq 1 ]; then
    sudo -n cp $L/libhsa-runtime64.so.1.18.0 /tmp/hsa.tmp && sudo -n mv /tmp/hsa.tmp $L/libhsa-runtime64.so.1.18.0 && sudo -n ldconfig
    python3 -c 'import os
for f in ["/opt/bc250-rocm/lib64/libhsa-runtime64.so.1.18.0","/usr/lib64/libamdhip64.so.7.1.52802"]:
    fd=os.open(f,os.O_RDONLY); os.fsync(fd) if False else None; os.posix_fadvise(fd,0,0,os.POSIX_FADV_DONTNEED); os.close(fd)'
    tag="relinked"
  else
    tag="untouched"
  fi
  timeout -k 30 1500 gdb -q -batch -x $O/gdb.cmds --args $M/build-hip-f32iq/bin/llama-bench -m /opt/models/qwen3-14b.gguf -ngl 99 -fa on -p 0 -n 64 -d 16384 -r 2 > $O/relink-r$i.log 2>&1
  if grep -q "SIGSEGV" $O/relink-r$i.log; then echo "run $i ($tag): SIGSEGV" >> $O/relink.log; else echo "run $i ($tag): $(grep -aoE 'received signal [A-Z]+' $O/relink-r$i.log | head -1)" >> $O/relink.log; fi
  sleep 45
done
echo RELINK DONE >> $O/relink.log
