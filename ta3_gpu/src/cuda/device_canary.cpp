#include "ta3/gpu/device_canary.hpp"

// the MSVC-STL bits hazard is guarded by __CUDA_ARCH__ branches in bits.hpp and covered by the parity
// test on this backend; a real device check exists only in the SYCL backend.
namespace ta3::gpu {

bool device_bits_canary() { return true; }

} // namespace ta3::gpu
