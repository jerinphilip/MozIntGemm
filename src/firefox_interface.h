#pragma once

#include <cstdint>

using Index = uint32_t;

namespace pg {
namespace Ruy {
#include "firefox_interface.inl"

namespace detail {

template <class Path> struct Preprocess;
struct kStandardCpp;
struct kNeon;

} // namespace detail

} // namespace Ruy

namespace Intgemm {
#include "firefox_interface.inl"
}
} // namespace pg
