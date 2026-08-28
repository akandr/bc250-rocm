#!/usr/bin/env bash
# Every harness that counts GPU faults must use an instrument that could see one.
#
# `dmesg` reads the current boot's ring buffer, and only what is still in it. If a
# run ends in a GPU reset the board goes down, and the dmesg check that follows
# reads the buffer of the boot after the crash, which is empty by construction.
# On a long run the buffer also wraps: a fault hunt flooding the log with SVM
# messages pushed the boot-time 40-CU unlock line out of dmesg entirely while it
# sat in the persistent journal the whole time. Harnesses in this repository
# reported "0 faults" that way for weeks while five fatal resets sat in the
# persistent journal.
#
# A harness may still use dmesg for a within-boot count, but it must also
# consult the journal, so this flags any script that counts faults and never
# mentions journalctl.
set -u
# Resolve to the repository root, as audit_logs.sh does, so the globs below
# cannot silently match nothing and report a clean result.
cd "$(dirname "$0")/.." || exit 1
bad=0
for f in scripts/*.sh reproduce.sh; do
  grep -qE "dmesg.*grep -c|grep -c.*dmesg" "$f" || continue
  grep -q "journalctl" "$f" && continue
  # a historical harness may instead state the limitation in its header
  if ! grep -q "fault_count.sh" "$f"; then
    echo "  COUNTS FAULTS WITH dmesg ALONE: $f"
    bad=$((bad+1))
  fi
done
echo "$bad script(s) counting GPU faults without consulting the persistent journal"

# Checking the instrument is not checking the query. This audit passed for a week
# while fault_count.sh, the instrument it points every harness at, used a pattern
# that matched no page fault on this board. Tested against a captured journal it
# found 62 preemption and flush lines and none of the three page faults that
# opened the fault. So also require that anything counting faults looks for the
# page-fault line, which is the class this hardware actually produces.
missing=0
for f in scripts/*.sh reproduce.sh; do
  # the quote may be backslash-escaped, since these calls sit inside \$( ) in a
  # double-quoted log line. The first version of this check missed every such
  # script for exactly that reason, which is the fault it exists to catch.
  grep -qE "grep -c[a-zA-Z]*E? *\\\\?[\"'][^\"']*(fault|preemption)" "$f" || continue
  # Mentioning the shared counter in a header is not using it, so require a real
  # call: a line that sources the file or invokes fault_counts. An earlier
  # version of this test accepted any mention and skipped every script, because
  # the advisory sentence in their headers reads ". Use scripts/fault_count.sh".
  grep -qE "^[[:space:]]*(\\.|source)[[:space:]]+.*fault_count\\.sh|(^|[^a-z_])fault_counts([^a-z_]|$)" "$f" && continue
  grep -qE "page fault" "$f" && continue
  echo "  COUNTS FAULTS WITHOUT MATCHING PAGE FAULTS: $f"
  missing=$((missing+1))
done
echo "$missing script(s) counting faults with a pattern that cannot match a page fault"
