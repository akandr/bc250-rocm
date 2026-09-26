// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
#include <hip/hip_runtime.h>
typedef int8_t int8x4_t __attribute__((ext_vector_type(4)));
__device__ int scalar_dp4a(int a,int b,int c){
  const int8x4_t va = reinterpret_cast<const int8x4_t&>(a);
  const int8x4_t vb = reinterpret_cast<const int8x4_t&>(b);
  c += va[0]*vb[0] + va[1]*vb[1] + va[2]*vb[2] + va[3]*vb[3];
  return c;
}
__device__ int rdna1_dp4a(int a,int b,int c){
  int t1,t2;
  asm("\n v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_0 src1_sel:BYTE_0 \n v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_1 src1_sel:BYTE_1 \n v_add3_u32 %0, %1, %2, %0 \n v_mul_i32_i24 %1, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_2 src1_sel:BYTE_2 \n v_mul_i32_i24 %2, sext(%3), sext(%4) dst_sel:DWORD dst_unused:UNUSED_PAD src0_sel:BYTE_3 src1_sel:BYTE_3 \n v_add3_u32 %0, %1, %2, %0 \n" : "+v"(c), "=&v"(t1), "=&v"(t2) : "v"(a), "v"(b));
  return c;
}
__global__ void ks(int*o,const int*a,const int*b){ *o = scalar_dp4a(a[threadIdx.x],b[threadIdx.x],0); }
__global__ void kr(int*o,const int*a,const int*b){ *o = rdna1_dp4a(a[threadIdx.x],b[threadIdx.x],0); }
