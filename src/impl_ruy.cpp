#include "firefox_interface.h"
#include "ruy/ruy.h"
#include <iostream>
#include <memory>

namespace pg::Ruy {
// The following is bad practice. But this makes it easy for @jerinphilip to
// export the cpp file directly for Mozilla as an implementation.
#include "impl_ruy-export.cpp"

} // namespace pg::Ruy
