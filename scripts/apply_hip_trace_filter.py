#!/usr/bin/env python3
# Gate the HIP runtime's API-call trace on a per-function filter, to bisect which traced call
# suppresses the zeroed fp16 GEMM (logs/fp16-scalar-2026-09-15/).
#
# Applied to hipamd/src/hip_internal.hpp of the rocclr 6.4.2 source (Fedora rocclr-6.4.2-2.fc43
# SRPM), which rebuilds to a libamdhip64.so.6.4.43484 that reproduces both the defect and its
# suppression by AMD_LOG_LEVEL=3 AMD_LOG_MASK=1.
#
#   BC250_APITRACE_FILTER unset      trace every API, as stock
#   BC250_APITRACE_FILTER=NONE       trace none through these macros
#   BC250_APITRACE_FILTER=a,b,c      trace only APIs whose name contains one of the substrings;
#                                    a token written =name matches that function name exactly
#   BC250_APITRACE_SITES=entry|exit|both (default both): which of the two print points run
#   BC250_APITRACE_ACTION=sleep:<us>|write|clock|stackfill|stackzero|nop  run at the entry point
#                                    of matched APIs (stackfill/zero write 16 KiB below the frame),
#                                    whatever AMD_LOG_LEVEL is
#
# Only the three macro print sites are gated (entry print, duration return print, error return
# print). Other LOG_API prints elsewhere in the runtime are left alone.
#
# Usage: apply_hip_trace_filter.py <clr-rocm-6.4.2>/hipamd/src/hip_internal.hpp
import sys
p = sys.argv[1]
s = open(p).read()
if "bc250_trace_match" in s:
    print("already patched"); sys.exit(0)

helper = r'''
#include <cstdlib>
#include <cstring>
#include <string>
#include <ctime>
#include <unistd.h>
// BC-250 experiment, see scripts/apply_hip_trace_filter.py
inline bool bc250_trace_match(const char* f, int site) {
  static const char* flt = getenv("BC250_APITRACE_FILTER");
  static const char* sites = getenv("BC250_APITRACE_SITES");
  if (sites != nullptr) {
    if (site == 0 && strcmp(sites, "exit") == 0) return false;
    if (site != 0 && strcmp(sites, "entry") == 0) return false;
  }
  if (flt == nullptr) return true;
  if (strcmp(flt, "NONE") == 0) return false;
  std::string s(flt);
  size_t start = 0;
  while (start <= s.size()) {
    size_t comma = s.find(',', start);
    std::string tok = s.substr(start, comma == std::string::npos ? std::string::npos : comma - start);
    if (!tok.empty() && tok[0] == '=' && strcmp(f, tok.c_str() + 1) == 0) return true;
    if (!tok.empty() && tok[0] != '=' && strstr(f, tok.c_str()) != nullptr) return true;
    if (comma == std::string::npos) break;
    start = comma + 1;
  }
  return false;
}
// BC250_APITRACE_ACTION: run instead of (not as well as) the entry print, for matched APIs,
// independent of AMD_LOG_LEVEL. sleep:<us> | write | clock | none
__attribute__((noinline)) inline void bc250_stackfill(int v) {
  volatile unsigned char buf[16384];
  for (size_t i = 0; i < sizeof(buf); i++) buf[i] = (unsigned char) v;
}
inline void bc250_trace_action(const char* f) {
  static const char* act = getenv("BC250_APITRACE_ACTION");
  if (act == nullptr || !bc250_trace_match(f, 0)) return;
  if (strncmp(act, "sleep:", 6) == 0) { usleep((useconds_t) atoi(act + 6)); return; }
  if (strcmp(act, "write") == 0) { static const char m[] = "bc250 action\n"; (void)!write(2, m, sizeof(m) - 1); return; }
  if (strcmp(act, "clock") == 0) { struct timespec ts; clock_gettime(CLOCK_MONOTONIC, &ts); return; }
  if (strcmp(act, "stackfill") == 0) { bc250_stackfill(0x55); return; }
  if (strcmp(act, "stackzero") == 0) { bc250_stackfill(0x00); return; }
  /* "nop": matched and returns, doing nothing */
}
'''
anchor = "#define HIP_API_PRINT(...)"
assert anchor in s
s = s.replace(anchor, helper + "\n" + anchor, 1)

old1 = '''#define HIP_API_PRINT(...)                                          \\
  uint64_t startTimeUs = 0;                                         \\
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs,       \\'''
new1 = '''#define HIP_API_PRINT(...)                                          \\
  uint64_t startTimeUs = 0;                                         \\
  bc250_trace_action(__func__);                                     \\
  if (bc250_trace_match(__func__, 0))                               \\
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs,       \\'''
assert old1 in s, "entry print anchor"
s = s.replace(old1, new1, 1)

old2 = '''#define HIP_ERROR_PRINT(err, ...)                                                  \\
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s : %s",                     \\'''
new2 = '''#define HIP_ERROR_PRINT(err, ...)                                                  \\
  if (bc250_trace_match(__func__, 1))                                              \\
  ClPrint(amd::LOG_INFO, amd::LOG_API, "%s: Returned %s : %s",                     \\'''
assert old2 in s, "error print anchor"
s = s.replace(old2, new2, 1)

old3 = '''  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs, "%s: Returned %s : %s", __func__,    \\'''
new3 = '''  if (bc250_trace_match(__func__, 2))                                                               \\
  HIPPrintDuration(amd::LOG_INFO, amd::LOG_API, &startTimeUs, "%s: Returned %s : %s", __func__,    \\'''
assert s.count(old3) == 1, "duration print anchor"
s = s.replace(old3, new3, 1)
open(p, "w").write(s)
print("patched", p)
