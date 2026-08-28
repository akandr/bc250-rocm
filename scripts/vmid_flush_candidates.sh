#!/usr/bin/env bash
# Can a KFD-scoped VMID flush replace the runlist rebuild?
#
# Established by instrumentation: the gfx10 PASID flush matches no VMID on this
# ASIC and silently does nothing, which is why the runlist rebuild exists. A
# first fallback invalidating all VMIDs 1..15 was tested and rejected: far too
# slow, and it hits graphics VMIDs too. This restricts it to the range the KFD
# owns, from vm_manager.first_kfd_vmid upward.
#
# The discriminating workload is a model load, not seq_probe: at runlist=0 a
# load aborts with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION, whereas seq_probe
# passes at runlist=0 and cannot tell the cells apart. That was a flaw in the
# previous attempt and is corrected here.
#
# Fault counts below come from dmesg, which sees only the current boot and
# only what is still in the ring buffer. If a run ends in a GPU reset the
# board goes down and the following check reads an empty buffer; and on a long
# run the buffer wraps, so early messages are gone. A zero here means "nothing
# dmesg can still see", not "nothing happened". Use scripts/fault_count.sh in
# new work.
set -u
D=~/inv82; mkdir -p "$D"
L=${ROCBLAS_LIB_DIR:-/home/akandr/rocBLAS/build/release/rocblas-install/lib}
HIP=~/llama-master/build-hip/bin
M8=/opt/models/qwen3-8b-q8_0.gguf
RL=/sys/module/amdgpu/parameters/bc250_flush_by_runlist
VM=/sys/module/amdgpu/parameters/bc250_flush_vmid_mode
log () { echo "[$(date +%H:%M:%S)] $*" | tee -a "$D/log"; sync; }

cell () { # cell <runlist> <vmid_mode> <label>
  echo $1 | sudo tee $RL > /dev/null; echo $2 | sudo tee $VM > /dev/null; sleep 1
  local t0=$(date +%s)
  env HSA_ENABLE_SDMA=0 GGML_CUDA_CUBLAS_COMPUTE_TYPE=f32 LD_LIBRARY_PATH=$L \
    timeout -k 20 420 $HIP/llama-bench -m $M8 -mmp 0 -ngl 99 -fa on -p 512 -n 64 -r 1 \
    > "$D/rl${1}_vm${2}.log" 2>&1
  local rc=$? t1=$(date +%s)
  local tg=$(grep -aoE "tg64 \| +[0-9.]+" "$D/rl${1}_vm${2}.log" | grep -oE "[0-9.]+$")
  local ap=$(grep -ac "APERTURE_VIOLATION\|aborting" "$D/rl${1}_vm${2}.log")
  log "  runlist=$1 vmid_mode=$2 ($3): rc=$rc tg64=${tg:-NONE} aperture_aborts=$ap wall=$((t1-t0))s"
}

log "=== does the model load survive each configuration?"
cell 3 0 "current recipe: runlist rebuild, no vmid flush"
cell 0 0 "no flush at all: expected to abort"
cell 0 2 "KFD-scoped vmid flush only: the candidate"
cell 0 1 "all-VMID flush only: known bad, for contrast"
log "=== repeat the two that matter, order reversed"
cell 0 2 "KFD-scoped again"
cell 0 0 "no flush again"

echo 3 | sudo tee $RL > /dev/null; echo 0 | sudo tee $VM > /dev/null
log "=== restored runlist=3 vmid_mode=0"
log "=== dmesg faults: $(sudo dmesg | grep -ciE \"memory access fault|preemption time out|APERTURE\")"
touch "$D/DONE"; log done
