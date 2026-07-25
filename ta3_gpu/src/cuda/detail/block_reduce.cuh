#pragma once

#include <ta3/ai/search/beam.hpp>

#include <cstdint>
#include <span>

/**
 * @file block_reduce.cuh
 * @brief block-wide reduction / top-K selection primitives shared by the beam kernel's stages.
 * @details structural twin of ../../sycl/detail/block_reduce.hpp -- same functions, same names; warp
 *  shuffles and @c __syncthreads() here, sub-group primitives and @c group_barrier there. scratch is
 *  sized to the warp count, not the block, because the warp width is fixed at 32 (the SYCL twin cannot
 *  assume that, so it sizes to the whole work-group).
 *
 * every array parameter is a fixed-extent std::span: the extent is a template argument, not stored, so
 * span<T,N> compiles to exactly a T* -- identical codegen to a raw pointer -- while the compiler now
 * rejects a caller passing a wrong-sized array. relies on --expt-relaxed-constexpr (already required by
 * this project for the search core, see CMakeLists.txt) to make span's constexpr member functions
 * device-callable.
 */
namespace ta3::gpu {
namespace {

    /**
     * block-wide max of per-thread packed keys, result returned to every thread (not just broadcast via a
     * shared scalar: each thread recombines the per-warp maxima itself, since WARPS is tiny (<=4 at both
     * call sites) that redundant work is cheaper than a second barrier).
     * warp shuffle reduction -> one slot per warp -> every thread combines the (<=4) warp maxima itself.
     * one barrier, no atomics.
     * @param warp_best scratch, [Block/32] entries.
     * @note the caller still owes a barrier before anything else may write @p warp_best again (the next
     *  round's shuffle-reduce, typically) -- this function only guarantees the *read* side is safe.
     */
    template<std::uint32_t Block> requires(Block % 32 == 0)
    __device__ unsigned long long block_max_bcast(
        unsigned long long v,
        std::span<unsigned long long, Block / 32> warp_best,
        std::uint32_t tid
    ) {
        static constexpr std::uint32_t WARPS = Block / 32;
        std::uint32_t const lane = tid & 31u;
        std::uint32_t const wid = tid >> 5;

#pragma unroll
        for(int o = 16; o > 0; o >>= 1) {
            unsigned long long const other = __shfl_down_sync(0xFFFFFFFFu, v, o);
            v = other > v ? other : v;
        }
        if(lane == 0)
            warp_best[wid] = v;
        __syncthreads();

        unsigned long long m = warp_best[0];
#pragma unroll
        for(std::uint32_t w = 1; w < WARPS; ++w)
            m = warp_best[w] > m ? warp_best[w] : m;
        return m;
    }

    /**
     * odd-even transposition sort, descending: N rounds, each a fixed alternating pattern of even/odd
     * compare-exchanges. every index is a literal after unrolling (@p round and @p j are both compile-time
     * once the outer loop is unrolled, so `round & 1u` and the inner bound fold to constants too) -- unlike
     * an insertion sort, there is no data-dependent index anywhere, so @p a has no reason to leave
     * registers. N rounds is the standard conservative bound (an element can move at most one slot per
     * round, and needs to move at most N-1 slots).
     */
    template<std::uint32_t N>
    __device__ void sort_desc_network(unsigned long long (&a)[N]) {
#pragma unroll
        for(std::uint32_t round = 0; round < N; ++round) {
#pragma unroll
            for(std::uint32_t j = round & 1u; j + 1 < N; j += 2) {
                unsigned long long const hi = a[j] > a[j + 1] ? a[j] : a[j + 1];
                unsigned long long const lo = a[j] > a[j + 1] ? a[j + 1] : a[j];
                a[j] = hi;
                a[j + 1] = lo;
            }
        }
    }

    /**
     * device twin of @c ta3::ai::search::select_top: @p k rounds of block argmax, identical result by
     * construction ("max value, lowest slot on ties"), but without a per-round full rescan of @p vals.
     * each thread packs and descending-sorts its own strided candidates into registers exactly once up
     * front (via a fixed sorting network, not a variable-index insertion sort), then every round it offers
     * @c local[0] -- always the same literal index -- to the block reduction. the thread whose candidate
     * actually won shifts its whole array down by one (again fixed indices only): ties can't collide
     * because @c pack_key folds the slot into the key, so at most one thread's offer can equal the round's
     * winning key.
     * @tparam Space the slot-value array's extent (compile-time -- ROOT_SLOTS / MID_SLOTS at the two
     *  call sites).
     * @tparam KCap  the @p sel_slot / @p sel_val extent (the widest beam over all levels sharing them).
     * @param k      the round count == this level's beam width; uniform across the block. @pre k <= KCap.
     * @return the (uniform) number selected; stops early when no non-zero value remains.
     */
    template<std::uint32_t Block, std::uint32_t Space, std::uint32_t KCap> requires(Block % 32 == 0)
    __device__ std::uint32_t select_top_block(
        std::span<std::uint32_t const, Space> vals,
        std::span<std::uint32_t, KCap> sel_slot,
        std::span<std::uint32_t, KCap> sel_val,
        std::span<unsigned long long, Block / 32> warp_best,
        std::uint32_t tid,
        std::uint32_t k
    ) {
        static constexpr std::uint32_t CAP = (Space + Block - 1) / Block;

        // pack this thread's strided candidates and sort them into descending registers, once per call.
        unsigned long long local[CAP];
#pragma unroll
        for(std::uint32_t i = 0; i < CAP; ++i) {
            std::uint32_t const slot = tid + i * Block;
            local[i] = (slot < Space && vals[slot] != 0)
                       ? ta3::ai::search::pack_key(vals[slot], slot, Space)
                       : 0ull;
        }
        sort_desc_network(local);

        std::uint32_t count = 0;
        for(; count < k; ++count) {
            unsigned long long const offer = local[0];
            unsigned long long const best = block_max_bcast<Block>(offer, warp_best, tid);
            if(best == 0)
                break;

            if(offer == best) { // unique: pack_key folds the slot in, so only the winner's offer can match
                // pop the consumed top: shift down by one, constant indices only (fully unrolled).
                // (i + 1 < CAP, not i < CAP - 1: CAP can be 1 and the unsigned "< 0" trips nvcc #186-D)
#pragma unroll
                for(std::uint32_t i = 0; i + 1 < CAP; ++i)
                    local[i] = local[i + 1];
                local[CAP - 1] = 0ull;
            }
            if(tid == 0) {
                std::uint32_t const slot = ta3::ai::search::key_pos(best, Space);
                sel_slot[count] = slot;
                sel_val[count] = static_cast<std::uint32_t>(best >> 32);
            }
            // protects warp_best against next round's write racing this round's readers, and publishes
            // this round's sel_slot/sel_val writes.
            __syncthreads();
        }
        return count;
    }

} // namespace
} // namespace ta3::gpu
