// hip_api_shim.cpp - LD_PRELOAD interposer for HIP API calls made by llama.cpp and rocBLAS.
//
// Written to find which HIP API family is involved in the zeroed fp16 GEMM, which disappears when
// the HIP runtime's API-call trace is on (AMD_LOG_LEVEL=3 AMD_LOG_MASK=1) and not with kernel-
// argument tracing (logs/fp16-scalar-2026-09-15/). The trace adds a formatted line and a timing
// measurement to every API call; this adds a chosen side effect to one family of calls at a time.
//
//   BC250_SHIM_FAMILIES  comma list of: dev, mem, copy, sync, err, launch, ptr, all
//   BC250_SHIM_ACTION    none   forward only (measures interposition itself)
//                        write  one write() of 64 bytes to /dev/null
//                        stderr one short line to stderr, the closest match to the trace
//                        yield  sched_yield()
//                        clock  two clock_gettime() calls, as the trace's duration does
//   BC250_SHIM_COUNT=1   print per-function call counts at exit
//
// Build on the board:
//   g++ -O2 -shared -fPIC -D__HIP_PLATFORM_AMD__ hip_api_shim.cpp -o libhipshim.so -ldl
// Use:
//   LD_PRELOAD=./libhipshim.so BC250_SHIM_FAMILIES=copy BC250_SHIM_ACTION=stderr <command>
#include <hip/hip_runtime_api.h>
// hip_ext.h does not compile under g++ (a template body names an undeclared pArgs), so the one
// prototype needed from it, hipExtModuleLaunchKernel, is spelled out in the wrapper below.
#include <dlfcn.h>
#include <fcntl.h>
#include <sched.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <unistd.h>
#include <atomic>
#include <map>
#include <mutex>
#include <string>

namespace {
enum { F_DEV = 1, F_MEM = 2, F_COPY = 4, F_SYNC = 8, F_ERR = 16, F_LAUNCH = 32, F_PTR = 64 };
int g_families = -1, g_action = 0, g_devnull = -1;
bool g_count = false;
std::mutex g_mu;
std::map<std::string, unsigned long> *g_counts = nullptr;

void init() {
  if (g_families != -1) return;
  const char *f = getenv("BC250_SHIM_FAMILIES");
  int fam = 0;
  if (f) {
    std::string s(f);
    auto has = [&](const char *w) { return s.find(w) != std::string::npos; };
    if (has("all")) fam = 0x7f;
    if (has("dev")) fam |= F_DEV;
    if (has("mem")) fam |= F_MEM;
    if (has("copy")) fam |= F_COPY;
    if (has("sync")) fam |= F_SYNC;
    if (has("err")) fam |= F_ERR;
    if (has("launch")) fam |= F_LAUNCH;
    if (has("ptr")) fam |= F_PTR;
  }
  const char *a = getenv("BC250_SHIM_ACTION");
  g_action = !a ? 0 : !strcmp(a, "write") ? 1 : !strcmp(a, "stderr") ? 2 : !strcmp(a, "yield") ? 3
           : !strcmp(a, "clock") ? 4 : 0;
  g_devnull = open("/dev/null", O_WRONLY);
  g_count = getenv("BC250_SHIM_COUNT") != nullptr;
  if (g_count) g_counts = new std::map<std::string, unsigned long>();
  g_families = fam;
}

void hook(int family, const char *name) {
  init();
  if (g_count) { std::lock_guard<std::mutex> l(g_mu); (*g_counts)[name]++; }
  if (!(g_families & family)) return;
  static const char buf[64] = "bc250 shim -------------------------------------------------\n";
  timespec ts;
  switch (g_action) {
  case 1: (void)!write(g_devnull, buf, sizeof(buf)); break;
  case 2: fprintf(stderr, "BC250SHIM %s\n", name); break;
  case 3: sched_yield(); break;
  case 4: clock_gettime(CLOCK_MONOTONIC, &ts); clock_gettime(CLOCK_MONOTONIC, &ts); break;
  default: break;
  }
}

template <typename F> F real(const char *name) {
  void *p = dlsym(RTLD_NEXT, name);
  if (!p) { fprintf(stderr, "BC250SHIM cannot resolve %s\n", name); abort(); }
  return reinterpret_cast<F>(p);
}

struct Report {
  ~Report() {
    if (!g_counts) return;
    for (auto &kv : *g_counts) fprintf(stderr, "BC250SHIMCOUNT %s %lu\n", kv.first.c_str(), kv.second);
  }
} g_report;
}  // namespace

#define W(fam, ret, name, sig, call)                                    \
  extern "C" ret name sig {                                             \
    static auto fn = real<ret(*) sig>(#name);                           \
    hook(fam, #name);                                                   \
    return fn call;                                                     \
  }

W(F_DEV, hipError_t, hipSetDevice, (int d), (d))
W(F_DEV, hipError_t, hipGetDevice, (int *d), (d))
W(F_DEV, hipError_t, hipDeviceGetAttribute, (int *pi, hipDeviceAttribute_t attr, int d), (pi, attr, d))
W(F_MEM, hipError_t, hipMalloc, (void **p, size_t s), (p, s))
W(F_MEM, hipError_t, hipFree, (void *p), (p))
W(F_MEM, hipError_t, hipMallocAsync, (void **p, size_t s, hipStream_t st), (p, s, st))
W(F_MEM, hipError_t, hipFreeAsync, (void *p, hipStream_t st), (p, st))
W(F_MEM, hipError_t, hipHostMalloc, (void **p, size_t s, unsigned int fl), (p, s, fl))
W(F_MEM, hipError_t, hipHostFree, (void *p), (p))
W(F_MEM, hipError_t, hipMemGetInfo, (size_t *f, size_t *t), (f, t))
W(F_COPY, hipError_t, hipMemcpy, (void *d, const void *s, size_t n, hipMemcpyKind k), (d, s, n, k))
W(F_COPY, hipError_t, hipMemcpyAsync, (void *d, const void *s, size_t n, hipMemcpyKind k, hipStream_t st), (d, s, n, k, st))
W(F_COPY, hipError_t, hipMemset, (void *d, int v, size_t n), (d, v, n))
W(F_COPY, hipError_t, hipMemsetAsync, (void *d, int v, size_t n, hipStream_t st), (d, v, n, st))
W(F_SYNC, hipError_t, hipStreamSynchronize, (hipStream_t st), (st))
W(F_SYNC, hipError_t, hipDeviceSynchronize, (void), ())
W(F_SYNC, hipError_t, hipStreamQuery, (hipStream_t st), (st))
W(F_SYNC, hipError_t, hipEventRecord, (hipEvent_t e, hipStream_t st), (e, st))
W(F_SYNC, hipError_t, hipEventSynchronize, (hipEvent_t e), (e))
W(F_SYNC, hipError_t, hipStreamIsCapturing, (hipStream_t st, hipStreamCaptureStatus *cs), (st, cs))
W(F_ERR, hipError_t, hipGetLastError, (void), ())
W(F_ERR, hipError_t, hipPeekAtLastError, (void), ())
W(F_PTR, hipError_t, hipPointerGetAttributes, (hipPointerAttribute_t *a, const void *p), (a, p))
W(F_LAUNCH, hipError_t, hipLaunchKernel, (const void *f, dim3 nb, dim3 db, void **args, size_t sm, hipStream_t st), (f, nb, db, args, sm, st))
W(F_LAUNCH, hipError_t, __hipPushCallConfiguration, (dim3 g, dim3 b, size_t sm, hipStream_t st), (g, b, sm, st))
W(F_LAUNCH, hipError_t, __hipPopCallConfiguration, (dim3 *g, dim3 *b, size_t *sm, hipStream_t *st), (g, b, sm, st))
W(F_LAUNCH, hipError_t, hipExtModuleLaunchKernel,
  (hipFunction_t f, uint32_t gx, uint32_t gy, uint32_t gz, uint32_t lx, uint32_t ly, uint32_t lz,
   size_t sm, hipStream_t st, void **kp, void **ex, hipEvent_t se, hipEvent_t ee, uint32_t fl),
  (f, gx, gy, gz, lx, ly, lz, sm, st, kp, ex, se, ee, fl))
