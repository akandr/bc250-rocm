// SPDX-License-Identifier: AGPL-3.0-or-later
// Copyright (c) 2026 Artur Andrzejczak <andrzejczak.artur@gmail.com>
// Assisted-by: Claude (Anthropic)
#include <hip/hip_runtime.h>
__global__ void k(int*o,const int*a,const int*b){ *o = __builtin_amdgcn_sdot4(a[0],b[0],0,false); }
