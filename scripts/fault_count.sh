#!/usr/bin/env bash
# Count GPU faults with an instrument that can see one across a reset.
#
# Source this and call fault_counts. It prints two numbers: faults in the
# current boot, and faults in the boot before it, which is where the evidence
# lives when a run ends by taking the board down.
# The pattern below missed page faults until 25 August, which is the fault class
# this board actually produces. Tested against the journal captured in
# logs/fault-usability-2026-08-24/: the old pattern matched 62 lines, all of them
# "Queue preemption failed" and "runlist rebuild flush failed", and none of the
# three "[gfxhub] page fault (src_id:...)" lines that opened the fault. Worse,
# "memory access fault" matches nothing on this board at all; it is a string the
# ROCr runtime prints to a process, not one the kernel logs. A page fault that
# does not go on to wedge a queue would have been counted as zero.
# Test any change to this against a captured journal rather than by reading it.
PAT="page fault \(src_id|memory access fault|preemption time out|Queue preemption failed|runlist rebuild flush failed|GPU reset begin"
fault_counts () {
  local now prev
  now=$(sudo journalctl -b 0 --no-pager 2>/dev/null | grep -ciE "$PAT")
  prev=$(sudo journalctl -b -1 --no-pager 2>/dev/null | grep -ciE "$PAT")
  echo "${now:-0} ${prev:-0}"
}
