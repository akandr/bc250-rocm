#!/usr/bin/env bash
# Summarise the KIQ repetition campaign into the table the write-up needs.
#
# One row per trial, so a configuration that splits across its repetitions is
# visible rather than averaged away. That is the whole point of the campaign: the
# observation it replaces was a single run, and single runs on this board have
# produced three convincing effects that reversed on resampling.
#
# Columns: which of the two read prints landed, and the highest step marker
# reached inside gfx_v10_0_kiq_init_queue(). "before read" landing with nothing
# after it means the CPU hung inside the register read itself.
set -eu
D=${1:-logs/reset-kiq-repeat-2026-08-24}
printf "| trial | before read | after read | highest step | last line |\n"
printf "|-------|-------------|------------|--------------|-----------|\n"
for f in "$D"/[ABCD]_rep*.log; do
	[ -e "$f" ] || continue
	n=$(basename "$f" .log)
	b=no; a=no
	grep -qa "BC250ACT before read" "$f" && b=yes
	grep -qa "BC250ACT after read" "$f" && a=yes
	s=$(grep -aoE "BC250ACT step [0-9]" "$f" | tail -1 | awk '{print $3}')
	[ -z "$s" ] && s=none
	l=$(grep -a . "$f" | tail -1 | sed 's/^\[[^]]*\] *//' | cut -c1-46)
	printf "| %s | %s | %s | %s | %s |\n" "$n" "$b" "$a" "$s" "$l"
done
