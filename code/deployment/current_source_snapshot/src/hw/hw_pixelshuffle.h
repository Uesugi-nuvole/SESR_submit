#ifndef __HW_PIXELSHUFFLE_H_
#define __HW_PIXELSHUFFLE_H_

#include "../ncnn/mat.h"

namespace accel {

class hw_pixelshuffle {
  public:
    int upscale_factor;
    int mode;

  public:
    hw_pixelshuffle();
    void forward(ncnn::Mat &ifm, ncnn::Mat &ofm);
};

} // namespace accel

#endif
