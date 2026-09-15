#!/usr/bin/env python3
# Record, without allocating, what ROCclr 6.4.2 captures for kernels with many parameters
# (Tensile GEMM kernels have far more than ggml's), and print it at process exit.
#
# Purpose: compare the captured launch of the zeroed fp16 GEMM (GLIBC_TUNABLES=
# glibc.malloc.tcache_count=1) with a correct one (tcache_count=0). The defect is sensitive to
# heap reuse, so the recorder writes only into a static buffer and prints from a destructor.
#
#   BC250_ARGDUMP=1   record launches whose kernel has >= BC250_ARGDUMP_MINPARAMS parameters
#                     (default 12), up to 64 launches
#
# For each parameter: index, type, size, the raw value captured (first 8 bytes), and for pointer
# parameters the global address passed and the memory object found for it (pointer and size),
# the parameter and type names from the kernel signature, and the host address the value was read from.
#
# Usage: apply_hip_argdump.py <clr-rocm-6.4.2>/rocclr/platform/kernel.cpp
import sys
p = sys.argv[1]
s = open(p).read()
if "bc250_argdump" in s:
    print("already patched"); sys.exit(0)

rec = r'''
// ---- BC-250 experiment, see scripts/apply_hip_argdump.py ----
#include <cstdio>
#include <cstdlib>
namespace {
struct Bc250ArgRec { uint32_t launch, idx, type, size; uint64_t value, global; void* mem; uint64_t memsize; const char* name; const char* tname; const void* src; };
static Bc250ArgRec bc250_recs[8192];
static uint32_t bc250_nrecs = 0, bc250_launches = 0;
static int bc250_argdump_state = -1, bc250_minparams = 12;
struct Bc250ArgDumpPrinter {
  ~Bc250ArgDumpPrinter() {
    if (bc250_nrecs == 0) return;
    for (uint32_t i = 0; i < bc250_nrecs; i++) {
      const Bc250ArgRec& r = bc250_recs[i];
      fprintf(stderr, "BC250ARG launch=%u idx=%u type=%u size=%u value=0x%016llx global=0x%016llx mem=%p memsize=%llu name=%s tname=%s src=%p\n",
              r.launch, r.idx, r.type, r.size, (unsigned long long) r.value, (unsigned long long) r.global,
              r.mem, (unsigned long long) r.memsize, r.name ? r.name : "?", r.tname ? r.tname : "?", r.src);
    }
  }
} bc250_argdump_printer;
}  // namespace
'''
anchor = "// =================================================================================================\nbool KernelParameters::captureAndSet("
assert anchor in s, "captureAndSet anchor"
s = s.replace(anchor, rec + anchor, 1)

old_loop_end = '''    desc.info_.defined_ = true;
  }

  execInfoOffset_ = totalSize_;
  return true;
}'''
assert old_loop_end in s, "loop end anchor"
new_loop_end = '''    desc.info_.defined_ = true;
    if (bc250_dump_this && bc250_nrecs < 8192) {
      Bc250ArgRec& r = bc250_recs[bc250_nrecs++];
      r.launch = bc250_launches; r.idx = (uint32_t) idx; r.type = (uint32_t) desc.type_;
      r.size = (uint32_t) desc.size_; r.value = 0; r.global = 0; r.mem = memArg; r.memsize = 0;
      r.name = desc.name_.c_str(); r.tname = desc.typeName_.c_str(); r.src = value;
      ::memcpy(&r.value, param, desc.size_ < 8 ? desc.size_ : 8);
      if (desc.type_ == T_POINTER && (desc.addressQualifier_ != CL_KERNEL_ARG_ADDRESS_LOCAL)) {
        r.global = (uint64_t) *reinterpret_cast<const void* const*>(value);
        if (memArg != nullptr) r.memsize = memArg->getSize();
      }
    }
  }
  if (bc250_dump_this) bc250_launches++;

  execInfoOffset_ = totalSize_;
  return true;
}'''
s = s.replace(old_loop_end, new_loop_end, 1)

old_head = '''bool KernelParameters::captureAndSet(void** kernelParams, address kernArgs, address mem) {
'''
new_head = '''bool KernelParameters::captureAndSet(void** kernelParams, address kernArgs, address mem) {
  if (bc250_argdump_state < 0) {
    const char* e = getenv("BC250_ARGDUMP");
    bc250_argdump_state = (e != nullptr) ? 1 : 0;
    const char* m = getenv("BC250_ARGDUMP_MINPARAMS");
    if (m != nullptr) bc250_minparams = atoi(m);
  }
  const bool bc250_dump_this = bc250_argdump_state == 1 && bc250_launches < 64 &&
                               signature_.numParameters() >= (size_t) bc250_minparams;
'''
assert old_head in s, "head anchor"
s = s.replace(old_head, new_head, 1)
open(p, "w").write(s)
print("patched", p)
