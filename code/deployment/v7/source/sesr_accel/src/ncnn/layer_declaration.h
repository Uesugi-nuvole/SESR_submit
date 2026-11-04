// Minimal layer declarations for the SESR simulation target.

#include "../layer/input.h"
namespace ncnn {
class Input_final : virtual public Input {};
DEFINE_LAYER_CREATOR(Input_final)
} // namespace ncnn

#include "../layer/split.h"
namespace ncnn {
class Split_final : virtual public Split {};
DEFINE_LAYER_CREATOR(Split_final)
} // namespace ncnn

#include "../layer/binaryop.h"
namespace ncnn {
class BinaryOp_final : virtual public BinaryOp {};
DEFINE_LAYER_CREATOR(BinaryOp_final)
} // namespace ncnn

#include "../layer/convolution.h"
namespace ncnn {
class Convolution_final : virtual public Convolution {};
DEFINE_LAYER_CREATOR(Convolution_final)
} // namespace ncnn

#include "../layer/padding.h"
namespace ncnn {
class Padding_final : virtual public Padding {};
DEFINE_LAYER_CREATOR(Padding_final)
} // namespace ncnn

#include "../layer/relu.h"
namespace ncnn {
class ReLU_final : virtual public ReLU {};
DEFINE_LAYER_CREATOR(ReLU_final)
} // namespace ncnn

#include "../layer/pixelshuffle.h"
namespace ncnn {
class PixelShuffle_final : virtual public PixelShuffle {};
DEFINE_LAYER_CREATOR(PixelShuffle_final)
} // namespace ncnn

#include "../layer/clip.h"
namespace ncnn {
class Clip_final : virtual public Clip {};
DEFINE_LAYER_CREATOR(Clip_final)
} // namespace ncnn
