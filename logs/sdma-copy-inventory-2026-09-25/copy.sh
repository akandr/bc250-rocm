#!/bin/bash
# Item 4: which transfer sizes does a decode actually issue? The SDMA engine beats the blit path
# just above 16 KiB and loses badly at 16 MiB, so where the copies land decides the cost.
export LC_ALL=C
cd /tmp || exit 1
hipcc -O2 -fPIC -shared copytrace.cpp -o libcopytrace.so -lroctracer64 -ldl 2>&1 | grep -iE "^[^ ].*error" | head -3
[ -f ./libcopytrace.so ] || { echo BUILD_FAILED; exit 1; }
B=/home/akandr/llama-master/build-hip-pkf16/bin
sleep 30
for m in qwen2.5-1.5b-q4km qwen3-8b-q8_0 qwen3.6-35b-a3b-iq2m; do
  for sdma in 0 1; do
    COPYTRACE_OUT=/tmp/copy_${m}_sdma${sdma}.txt LD_PRELOAD=/tmp/libcopytrace.so \
      HSA_ENABLE_SDMA=$sdma $B/llama-bench -m /opt/models/$m.gguf -mmp 0 -ngl 99 -fa on \
      -p 0 -n 64 -r 1 > /tmp/bench_${m}_sdma${sdma}.txt 2>/dev/null
    tg=$(grep -aoE "[0-9]+\.[0-9]+ ± " /tmp/bench_${m}_sdma${sdma}.txt | head -1)
    echo "== $m sdma=$sdma tg64=${tg:-?}"
    head -3 /tmp/copy_${m}_sdma${sdma}.txt
    sleep 15
  done
done
echo DONE_COPY
