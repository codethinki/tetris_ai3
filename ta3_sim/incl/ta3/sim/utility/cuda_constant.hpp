#pragma once

// __device__ (global memory, L1-cached), NOT __constant__: the constant bank broadcasts only when all
// lanes read the same address. these LUTs (placements, piece columns, plan trees) are indexed per lane,
// and divergent constant reads serialise into up-to-32-way replays; through L1 they are ordinary cached
// loads. the tables are a few hundred bytes and hot, so they stay resident in L1.
#if defined(__CUDACC__)
#  define TA3_CUDA_CONSTANT __device__ constexpr
#else
#  define TA3_CUDA_CONSTANT inline constexpr
#endif
