# Tetris Ai 3
### Introduction
This project trains a tetris ai on with a depth 3 beam search algorithm and customizable architecture.

The default small model 97 param model (v11) already achieves a near perfect score (only tetrises) after 20 iterations (200 generations)


### Showcase
The 4 Islands champions after ~20 iterations using the default model (200 generations with 4 islands, population of 50 with 50 games each) 
https://github.com/user-attachments/assets/00136c72-7955-40dc-ba97-96c470c4e049


### Build & Run
- The project compiles on windows & linux with gcc or msvc and uses vcpkg. 
- Additionally it requires to be pointed at the install output of (cth)[https://github.com/codethinki/cth]
- For compiling with the sycl backend, the sycl compiler is needed.
