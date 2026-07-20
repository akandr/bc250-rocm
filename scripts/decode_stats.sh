#!/bin/bash
# Solid decode statistics: N fresh-process decode attempts in one boot, classify each.
# Assumes fast loads (fence patch). Writes per-run verdict + running tally to ~/decode_stats.txt
N="${1:-25}"
export LD_LIBRARY_PATH=$HOME/llama.cpp/build-hip/bin:$LD_LIBRARY_PATH
u=$(cut -d. -f1 /proc/uptime); [ $u -lt 130 ] && sleep $((130-u))
sudo -n systemctl stop ollama queue-runner 2>/dev/null; sleep 1
OUT=$HOME/decode_stats.txt; : > $OUT
CB=0; FC=0; CLEAN=0; TMO=0; OTH=0
echo "### DECODE STATS N=$N $(date +%T) CU=$(journalctl -k -b 0|grep -o active_cu_number.[0-9]*|tail -1)" | tee -a $OUT
for i in $(seq 1 $N); do
  F=/tmp/ds.txt
  t0=$(date +%s)
  echo "Q: capital of France? A:" | env HSA_ENABLE_SDMA=0 GGML_CUDA_FORCE_MMQ=1 AMD_LOG_LEVEL=3 \
    ~/bin/timeout 150 ~/llama.cpp/build-hip/bin/llama-cli -m /opt/models/qwen2.5-1.5b-q4km.gguf \
    -ngl 999 -fa on --no-mmap -n 6 > $F 2>&1
  rc=$?; t1=$(date +%s)
  compute=$(grep -acE "ShaderName : void|ShaderName : mul_mat" $F)
  fault=$(grep -acE "APERTURE_VIOLATION|aborting with error" $F)
  lastk=$(awk '/ShaderName :/{k=$NF} /aborting with error|APERTURE_VIOLATION/{print k; exit}' $F)
  if [ "$fault" -gt 0 ]; then
    case "$lastk" in *copyBuffer*) v="FAULT_COPYBUFFER"; CB=$((CB+1));;
      *mul_mat*|*void*) v="FAULT_COMPUTE($lastk)"; FC=$((FC+1));;
      *) v="FAULT_$lastk"; FC=$((FC+1));; esac
  elif [ "$compute" -gt 200 ]; then v="RAN_CLEAN(k=$compute)"; CLEAN=$((CLEAN+1))
  elif [ "$rc" = 124 ]; then v="LOAD_TIMEOUT(k=$compute)"; TMO=$((TMO+1))
  else v="OTHER(rc=$rc k=$compute)"; OTH=$((OTH+1)); fi
  echo "run $i [$((t1-t0))s]: $v" | tee -a $OUT
  rm -f $F
  sleep 2
done
echo "### TALLY: clean=$CLEAN fault_copyBuffer=$CB fault_compute=$FC load_timeout=$TMO other=$OTH / $N" | tee -a $OUT
