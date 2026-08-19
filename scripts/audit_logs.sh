#!/usr/bin/env bash
# Check that every log directory documents itself.
#
# The write-up tells readers that each directory under logs/ has a README naming
# what produced it. That claim was not true for a third of them until it was
# checked, which is the reason this exists: a documentation promise is as easy to
# drift from as a number.
#
# Two things are checked: that a README exists, and that it says how the data was
# produced, whether by a named harness in scripts/ or by commands run directly.
set -u
cd "$(dirname "$0")/.." || exit 1
miss_readme=0; miss_method=0; total=0
for d in logs/*/; do
    total=$((total + 1))
    if [ ! -f "$d/README.md" ]; then
        echo "no README:  $d"; miss_readme=$((miss_readme + 1)); continue
    fi
    if ! grep -qiE "scripts/|harness|generated|produced|collected|reproduce|built |run as|run by|run with|invocation|command" "$d/README.md"; then
        echo "no method:  $d"; miss_method=$((miss_method + 1))
    fi
done
echo
echo "$total log directories: $miss_readme without a README, $miss_method without a method"
