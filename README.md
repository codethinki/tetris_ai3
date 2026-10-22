[![Build & Test](https://github.com/codethinki/tetris_ai3/actions/workflows/build-test.yml/badge.svg)](https://github.com/codethinki/tetris_ai3/actions/workflows/build-test.yml)

# Tetris Ai 3
trains a tetris ai with a depth 3 beam search and customizable model architecture

the default small 97 param model (v11) already achieves a near perfect score (only tetrises) after 20 iterations (200 generations)

built and tested with:
- windows: msvc, cuda 13

# showcase
the 4 island champions after ~20 iterations using the default model (200 generations with 4 islands, population of 50 with 50 games each)

https://github.com/user-attachments/assets/00136c72-7955-40dc-ba97-96c470c4e049

# overview

## ta3::sim
tetris simulation, constexpr

- `Board2`: bit column board, piece fitting, drops and line clears
- `TetrisEngine`: seeded 7-bag, hold, hard drop placements, one seed reproduces a whole game
- `TetrisGame`: move level layer: moves, rotations with wall kicks, hold, place
- `TetrisState`: everything a move search needs
- raylib renderer

## ta3::ai
models and search

- models `v7` - `v11`: small fully connected nets over board metrics, `v11` is 10 -> 8 -> 1
- metrics: board metrics fused into a single pass, stateful game metrics like the max height
- search: exhaustive and beam search over the placements of the current, held and next pieces
- preview: watch the current champions play

## ta3::trainer
training on [pagmo2](https://github.com/esa/pagmo2)

- island model (pagmo archipelago) with `DD_CMAES` (default) or `SEP_CMAES`
- candidates are scored by playing games on the gpu (`ta3::gpu`)
- checkpoint every iteration, keeps the last backups

## ta3::gpu
plays games for the trainer on the gpu, one backend per build (`TA3_GPU_BACKEND`)

- `CUDA`: nvcc
- `SYCL`: [AdaptiveCpp](https://github.com/AdaptiveCpp/AdaptiveCpp), built as a subproject with its own clang

# models
`models/` holds trained champions

# building

## requirements
- cmake 4.1.1+, ninja, [buildcache](https://gitlab.com/bits-n-bites/buildcache)
- [vcpkg](https://github.com/microsoft/vcpkg) with `VCPKG_ROOT` set
- msvc
- cuda backend: cuda toolkit 13.2+
- sycl backend: AdaptiveCpp, python on `PATH`

## build & test
```bash
git clone --recursive https://github.com/codethinki/tetris_ai3.git
cd tetris_ai3
cmake --preset msvc_release_cuda
cmake --build out/build/msvc_release_cuda
ctest --test-dir out/build/msvc_release_cuda
```

## presets
- cuda: `msvc_debug_cuda`, `msvc_reldebug_cuda`, `msvc_release_cuda`
- sycl: `msvc_reldebug_sycl`, `msvc_release_sycl`, set the AdaptiveCpp path (`TA3_ACPP_ROOT`) in `CMakeUserPresets.json`

## dependencies
- [cth](https://github.com/codethinki/cth) and [cth_cmake](https://github.com/codethinki/cth_cmake): submodules in `lib/`
- `TA3_CTH_DIR` points the build at a local cth clone to work on both

# training
`ta3_trainer_runner` asks for a model path, trains until enter is pressed and checkpoints every iteration

the settings (islands, population, games per evaluation, algorithm) are constants in `ta3_trainer/runner/src/main.cpp`

# license
[MIT](LICENSE)
