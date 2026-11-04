#include "hw_pixelshuffle.h"
#include "accel.h"
#include "accel_params.h"

namespace accel {

hw_pixelshuffle::hw_pixelshuffle() {
    this->upscale_factor = 2;
    this->mode = 0;
}







void hw_pixelshuffle::forward(ncnn::Mat &ifm, ncnn::Mat &ofm) {


    PIXELSHUFFLE_RESET;

    Xil_DCacheFlush();


    PIXELSHUFFLE_IFM_ADDR_SET((uint32_t)ifm.data);
    PIXELSHUFFLE_OFM_ADDR_SET((uint32_t)ofm.data);
    PIXELSHUFFLE_IFM_WH_SET(ifm.w, ifm.h);
    PIXELSHUFFLE_IFM_C_SET(ifm.c);
    PIXELSHUFFLE_IFM_CSTEP_SET(ifm.cstep);
    PIXELSHUFFLE_OFM_WH_SET(ofm.w, ofm.h);
    PIXELSHUFFLE_OFM_C_SET(ofm.c);
    PIXELSHUFFLE_OFM_CSTEP_SET(ofm.cstep);
    PIXELSHUFFLE_CTRL_SET(1, upscale_factor, mode);

    wait_pixelshuffle_done();

    Xil_DCacheInvalidate();


    PIXELSHUFFLE_RESET;
}

} // namespace accel
