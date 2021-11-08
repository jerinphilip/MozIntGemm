#pragma once

#include <cstdint>

using Index = uint32_t;

namespace pg {
namespace Ruy {
#include "ffox_intgemm.inl"
}

namespace Intgemm {
#include "ffox_intgemm.inl"
}
} // namespace pg
