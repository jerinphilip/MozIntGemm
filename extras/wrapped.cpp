#include "wrapped.h"

namespace pg::Ruy {

#include "MozIntGemm/moz_intgemm_ruy.inl"

} // namespace pg::Ruy

#if RUY_PLATFORM_X86
namespace pg::Intgemm {

#include "MozIntGemm/moz_intgemm_intgemm.inl"

} // namespace pg::Intgemm
#endif
