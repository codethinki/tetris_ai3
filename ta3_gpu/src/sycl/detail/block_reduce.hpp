#pragma once

#include <ta3/ai/search/beam.hpp>

#include <sycl/sycl.hpp>

#include <cstdint>
#include <span>

/**
 * @file block_reduce.hpp
 * @brief work-group-wide reduction / top-K selection primitives shared by the beam kernel's stages.
 * @details scratch is sized to the whole work-group, not Block/32: SYCL sub-group width is
 *  backend-dependent (not guaranteed 32-wide), unlike CUDA's fixed warp size.
 */
namespace ta3::gpu {
namespace {

    /**
     * work-group-wide max of per-work-item packed keys, result returned to every work-item: each work-item
     * recombines the per-sub-group maxima itself (cheaper than a second barrier since sub-group count is
     * small). one barrier, no atomics.
     * @param warp_best scratch, [Block] entries (sized to the work-group -- see the file-level note).
     * @note the caller still owes a barrier before anything else may write @p warp_best again -- this
     *  function only guarantees the *read* side is safe.
     */
    template<std::uint32_t Block>
    [[nodiscard]] inline unsigned long long block_max_bcast(
        unsigned long long v,
        sycl::nd_item<1> it,
        std::span<unsigned long long, Block> warp_best
    ) {
        sycl::sub_group const sg = it.get_sub_group();
        unsigned long long const mine = sycl::reduce_over_group(sg, v, sycl::maximum<unsigned long long>{});

        std::uint32_t const wid = static_cast<std::uint32_t>(sg.get_group_linear_id());
        std::uint32_t const numSubgroups = static_cast<std::uint32_t>(sg.get_group_linear_range());

        if(sg.leader())
            warp_best[wid] = mine;
        sycl::group_barrier(it.get_group());

        unsigned long long m = warp_best[0];
        for(std::uint32_t w = 1; w < numSubgroups; ++w)
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
    inline void sort_desc_network(unsigned long long (&a)[N]) {
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
     * device twin of @c ta3::ai::search::select_top: @p k rounds of work-group argmax, identical result by
     * construction ("max value, lowest slot on ties"), but without a per-round full rescan of @p vals.
     * each work-item packs and descending-sorts its own strided candidates into registers exactly once up
     * front (via a fixed sorting network, not a variable-index insertion sort), then every round it offers
     * @c local[0] -- always the same literal index -- to the work-group reduction. the work-item whose
     * candidate actually won shifts its whole array down by one (again fixed indices only): ties can't
     * collide because @c pack_key folds the slot into the key, so at most one work-item's offer can equal
     * the round's winning key.
     * @tparam Block work-group size (compile-time; @c BLOCK at both call sites).
     * @tparam Space the slot-value array's extent (compile-time -- ROOT_SLOTS / MID_SLOTS at the two
     *  call sites).
     * @tparam KCap  the @p sel_slot / @p sel_val extent (the widest beam over all levels sharing them).
     * @param k      the round count == this level's beam width; uniform across the work-group.
     *  @pre k <= KCap.
     * @return the (uniform) number selected; stops early when no non-zero value remains.
     */
    template<std::uint32_t Block, std::uint32_t Space, std::uint32_t KCap>
    [[nodiscard]] inline std::uint32_t select_top_block(
        std::span<std::uint32_t const, Space> vals,
        std::span<std::uint32_t, KCap> sel_slot,
        std::span<std::uint32_t, KCap> sel_val,
        std::span<unsigned long long, Block> warp_best,
        sycl::nd_item<1> it,
        std::uint32_t tid,
        std::uint32_t k
    ) {
        static constexpr std::uint32_t CAP = (Space + Block - 1) / Block;

        // pack this work-item's strided candidates and sort them into descending registers, once per call.
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
            unsigned long long const best = block_max_bcast<Block>(offer, it, warp_best);
            if(best == 0)
                break;

            if(offer == best) { // unique: pack_key folds the slot in, so only the winner's offer can match
                // pop the consumed top: shift down by one, constant indices only (fully unrolled).
#pragma unroll
                for(std::uint32_t i = 0; i < CAP - 1; ++i)
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
            sycl::group_barrier(it.get_group());
        }
        return count;
    }

} // namespace
} // namespace ta3::gpu
