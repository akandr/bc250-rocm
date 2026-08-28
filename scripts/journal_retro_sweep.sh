#!/usr/bin/env bash
# Ask the persistent journal what dmesg could not answer.
#
# A dmesg check run after a board reset reports the boot that followed the
# crash, so it cannot see the fault that caused it. This sweeps every retained
# boot for the GPU fault and reset signatures instead, and records the kernel
# release and scheduler policy of each so that boots can be identified without
# relying on wall-clock time, which jumps across reboots on this board.
set -u
for b in $(seq -19 0); do
  n=$(sudo journalctl -b "$b" --no-pager 2>/dev/null | grep -cE \
      "page fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin|signal 7/BUS")
  k=$(sudo journalctl -b "$b" --no-pager 2>/dev/null | grep -m1 -oE "vmlinuz-[0-9.]+-[0-9]+" | grep -oE "[0-9]+\.[0-9]+\.[0-9]+")
  sp=$(sudo journalctl -b "$b" --no-pager 2>/dev/null | grep -m1 -oE "sched_policy=[0-9]")
  kinds=$(sudo journalctl -b "$b" --no-pager 2>/dev/null | grep -oE \
      "page fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin|signal 7/BUS" \
      | sort | uniq -c | tr -s " " | tr "\n" ";")
  echo "boot=$b kernel=${k:-?} ${sp:-sched_policy=default} events=$n ${kinds}"
done
