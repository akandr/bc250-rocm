#!/usr/bin/env bash
# sdma_firmware_identity.sh - what the substituted SDMA blob actually is.
#
# The firmware page said the board's cyan_skillfish2 SDMA blob and the navi12
# one that replaces it are "the same SDMA 5.0 format, differing only in the
# ucode version field", from a reading that was not kept. This captures the
# comparison, including a byte diff, because "differing only in" is a much
# stronger claim than a header comparison can support.
#
# The board's own blobs are overwritten by the substitution, so the originals
# come from the backup the recipe tells you to take.
#
# AMD common_firmware_header, little-endian:
#   u32 size_bytes, u32 header_size_bytes, u16 header_version_major/minor,
#   u16 ip_version_major/minor, u32 ucode_version, u32 ucode_size_bytes,
#   u32 ucode_array_offset_bytes, u32 crc32
set -u
D="${1:-$HOME/sdma-identity}"
ORIG="${2:-$HOME/usr/lib/firmware/amdgpu}"
mkdir -p "$D"
FW=/lib/firmware/amdgpu

hdr () { xzcat "$1" | python3 -c '
import sys, struct
n = ["size_bytes","header_size_bytes","hdr_major","hdr_minor",
     "ip_major","ip_minor","ucode_version","ucode_size_bytes",
     "ucode_array_offset","crc32"]
for k, v in zip(n, struct.unpack("<IIHHHHIIII", sys.stdin.buffer.read(32))):
    print(f"  {k:20} 0x{v:x} ({v})")
'; }

{
  echo "date: $(date -Is)"
  echo "kernel: $(uname -r)"
  echo
  echo "=== in place at $FW (post-substitution) ==="
  for f in cyan_skillfish2_sdma.bin.xz navi12_sdma.bin.xz; do
    printf "%-34s compressed %s\n" "$f" "$(md5sum "$FW/$f" | cut -d' ' -f1)"
    printf "%-34s raw        %s\n" "" "$(xzcat "$FW/$f" | md5sum | cut -d' ' -f1)"
  done
  echo
  echo "=== the board's original, from the backup at $ORIG ==="
  for f in cyan_skillfish2_sdma.bin.xz cyan_skillfish2_sdma1.bin.xz; do
    [ -f "$ORIG/$f" ] || { echo "MISSING $f"; continue; }
    printf "%-34s raw        %s\n" "$f" "$(xzcat "$ORIG/$f" | md5sum | cut -d' ' -f1)"
  done
  echo
  echo "=== header: board original (cyan_skillfish2) ==="
  hdr "$ORIG/cyan_skillfish2_sdma.bin.xz"
  echo "=== header: navi12, the substitute ==="
  hdr "$FW/navi12_sdma.bin.xz"
  echo
  echo "=== byte comparison of the decompressed blobs ==="
  xzcat "$ORIG/cyan_skillfish2_sdma.bin.xz" > "$D/.orig.bin"
  xzcat "$FW/navi12_sdma.bin.xz"            > "$D/.navi.bin"
  echo "  size original: $(stat -c%s "$D/.orig.bin")   size navi12: $(stat -c%s "$D/.navi.bin")"
  echo "  bytes differing: $(cmp -l "$D/.orig.bin" "$D/.navi.bin" | wc -l)"
  echo "  first ten differing offsets (1-indexed, octal values):"
  cmp -l "$D/.orig.bin" "$D/.navi.bin" | head -10 | sed 's/^/    /'
  rm -f "$D/.orig.bin" "$D/.navi.bin"
} > "$D/identity.txt" 2>&1
