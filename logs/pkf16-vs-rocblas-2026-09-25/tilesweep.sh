#!/bin/bash
# Item 2: how much of the distance left to the 13.02 TFLOP/s ceiling is tile choice?
# Sweep the hand-written kernel's geometry at N=8192 and see where it plateaus.
export LC_ALL=C
cd /tmp || exit 1
echo "BM BN BK TM TN  LDS_B  VGPR  TFLOPs"
for cfg in "128 128 32 8 16" "128 128 16 8 16" "128 128 64 8 16" \
           "128 256 32 8 16" "256 128 32 8 16" "128 128 32 8 8" \
           "128 128 32 16 16" "64 128 32 8 16" "128 128 32 4 16"; do
  set -- $cfg; BM=$1; BN=$2; BK=$3; TM=$4; TN=$5
  NT=$(( (BM/TM) * (BN/TN) ))
  [ $NT -lt 32 ] && continue
  [ $NT -gt 1024 ] && continue
  out=$(hipcc -O3 --offload-arch=gfx1013 -DBM=$BM -DBN=$BN -DBK=$BK -DTM=$TM -DTN=$TN \
        --save-temps -c pk_gemm_square.cpp -o /tmp/t.o 2>&1)
  vg=$(grep -aoE "\.vgpr_count: *[0-9]+" /tmp/pk_gemm_square-hip-amdgcn-amd-amdhsa-gfx1013.s 2>/dev/null | head -1 | grep -oE "[0-9]+")
  ld=$(grep -aoE "\.group_segment_fixed_size: *[0-9]+" /tmp/pk_gemm_square-hip-amdgcn-amd-amdhsa-gfx1013.s 2>/dev/null | head -1 | grep -oE "[0-9]+")
  hipcc -O3 --offload-arch=gfx1013 -DBM=$BM -DBN=$BN -DBK=$BK -DTM=$TM -DTN=$TN \
        -o /tmp/pk_t pk_gemm_square.cpp -L/usr/lib64 -lamdhip64 2>/dev/null
  if [ -x /tmp/pk_t ]; then
    tf=$(/tmp/pk_t 2>/dev/null | awk '/square 8192/{print $3}')
  else tf="build-fail"; fi
  printf "%3s %3s %2s %2s %2s  %6s %5s  %s\n" $BM $BN $BK $TM $TN "${ld:-?}" "${vg:-?}" "${tf:-fail}"
  rm -f /tmp/pk_t
  sleep 20
done
echo DONE_TILE
