#include "hw_math.h"
#include "accel_params.h"
#include "accel.h"
namespace accel {

hw_math::hw_math() {
    this->mul_src1_sel = 0;
    this->add_src0_sel = 0;
    this->add_src1_sel = 0;
    this->sub_en = 0;
    this->add_en = 0;
    this->mul_en = 0;
    this->max_en = 0;
    this->min_en = 0;
    this->op = 0;
    this->alpha = 0.f;
    this->beta = 0.f;
    this->src1_broadcast = 0;
}

void hw_math::math_forward(ncnn::Mat &src0, ncnn::Mat &src1, ncnn::Mat &dst) {
    ALU_RESET;
    //Xil_DCacheInvalidate();
    int *src0_addr0 = src0;
    int *src1_addr0 = src1;
    int *dst_addr0 = dst;
    int len = src0.total();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = src0_addr0 + len_ch0;
    int *src1_addr1 = src1_addr0 + len_ch0;
    int *dst_addr1 = dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint32_t)src0_addr0);
    ALU_SRC1_ADDR_CH0_SET((uint32_t)src1_addr0);
    ALU_DST_ADDR_CH0_SET((uint32_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint32_t)src0_addr1);
    ALU_SRC1_ADDR_CH1_SET((uint32_t)src1_addr1);
    ALU_DST_ADDR_CH1_SET((uint32_t)dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,2,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,src1_broadcast,op);

    wait_alu_mat_done();
    //Xil_DCacheInvalidate();
    ALU_RESET;
}

void hw_math::math_forward_broadcast_src1(ncnn::Mat &src0, ncnn::Mat &src1, ncnn::Mat &dst) {
    if (!(src1.c == 1 && src0.w == src1.w && src0.h == src1.h)) {
        math_forward(src0, src1, dst);
        return;
    }

    const int plane = src0.w * src0.h;
    const int len_ch0 = (plane / 2 + 7) & -8;
    const int len_ch1 = u_align(plane - len_ch0, 4);

    for (int q = 0; q < src0.c; q++) {
        ALU_RESET;
        ////Xil_DCacheInvalidate();

        int *src0_ch0 = (int*)src0.channel(q).data;
        int *src1_ch0 = (int*)src1.channel(0).data;
        int *dst_ch0 = (int*)dst.channel(q).data;
        int *src0_ch1 = src0_ch0 + len_ch0;
        int *src1_ch1 = src1_ch0 + len_ch0;
        int *dst_ch1 = dst_ch0 + len_ch0;

        ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
        ALU_MATH_BETA_SET(*((uint32_t *)&beta));

        ALU_VECLEN_CH0_SET(len_ch0);
        ALU_SRC0_ADDR_CH0_SET((uint32_t)src0_ch0);
        ALU_SRC1_ADDR_CH0_SET((uint32_t)src1_ch0);
        ALU_DST_ADDR_CH0_SET((uint32_t)dst_ch0);

        ALU_VECLEN_CH1_SET(len_ch1);
        ALU_SRC0_ADDR_CH1_SET((uint32_t)src0_ch1);
        ALU_SRC1_ADDR_CH1_SET((uint32_t)src1_ch1);
        ALU_DST_ADDR_CH1_SET((uint32_t)dst_ch1);

        ALU_MATHFUNC_CTRL_SET(1,2,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,0,op);

        wait_alu_mat_done();
        //Xil_DCacheInvalidate();
        ALU_RESET;
    }
}

void hw_math::math_forward(ncnn::Mat &src0, ncnn::Mat &dst) {
    ALU_RESET;
    //Xil_DCacheInvalidate();
    int *src0_addr0 = src0;
    int *dst_addr0 = dst;
    int len = src0.total();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = src0_addr0 + len_ch0;
    int *dst_addr1 = dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint32_t)src0_addr0);
    ALU_DST_ADDR_CH0_SET((uint32_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint32_t)src0_addr1);
    ALU_DST_ADDR_CH1_SET((uint32_t)dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,1,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,0,op);

    wait_alu_mat_done();
    //Xil_DCacheInvalidate();
    ALU_RESET;
}

void hw_math::math_forward(void* src0_addr0, void* src1_addr0, void* dst_addr0, int len) {
    ALU_RESET;
    //Xil_DCacheInvalidate();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = (int*)src0_addr0 + len_ch0;
    int *src1_addr1 = (int*)src1_addr0 + len_ch0;
    int *dst_addr1 = (int*)dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint32_t)src0_addr0);
    ALU_SRC1_ADDR_CH0_SET((uint32_t)src1_addr0);
    ALU_DST_ADDR_CH0_SET((uint32_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint32_t)src0_addr1);
    ALU_SRC1_ADDR_CH1_SET((uint32_t)src1_addr1);
    ALU_DST_ADDR_CH1_SET((uint32_t)dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,2,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,src1_broadcast,op);

    wait_alu_mat_done();
    //Xil_DCacheInvalidate();
    ALU_RESET;
}

void hw_math::math_forward_broadcast_src1(void* src0_addr0, void* src1_addr0, void* dst_addr0, int w, int h, int c, int src1_w, int src1_h, int src1_c) {
    int plane = w * h;

    if (!(src1_c == 1 && w == src1_w && h == src1_h)) {
        math_forward(src0_addr0, src1_addr0, dst_addr0, w * h * c);
        return;
    }

    for (int q = 0; q < c; q++) {
        ALU_RESET;
        //Xil_DCacheFlush();

        int *src0_ch = (int*)src0_addr0 + q * plane;
        int *dst_ch = (int*)dst_addr0 + q * plane;
        int len_ch0 = (plane / 2 + 7) & -8;
        int len_ch1 = u_align(plane - len_ch0, 4);
        int *src0_ch1 = src0_ch + len_ch0;
        int *src1_ch1 = (int*)src1_addr0 + len_ch0;
        int *dst_ch1 = dst_ch + len_ch0;

        ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
        ALU_MATH_BETA_SET(*((uint32_t *)&beta));

        ALU_VECLEN_CH0_SET(len_ch0);
        ALU_SRC0_ADDR_CH0_SET((uint32_t)src0_ch);
        ALU_SRC1_ADDR_CH0_SET((uint32_t)src1_addr0);
        ALU_DST_ADDR_CH0_SET((uint32_t)dst_ch);

        ALU_VECLEN_CH1_SET(len_ch1);
        ALU_SRC0_ADDR_CH1_SET((uint32_t)src0_ch1);
        ALU_SRC1_ADDR_CH1_SET((uint32_t)src1_ch1);
        ALU_DST_ADDR_CH1_SET((uint32_t)dst_ch1);

        ALU_MATHFUNC_CTRL_SET(1,2,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,0,op);

        wait_alu_mat_done();
       // Xil_DCacheInvalidate();
        ALU_RESET;
    }
}

void hw_math::math_forward(void* src0_addr0, void* dst_addr0, int len) {
    ALU_RESET;
    //Xil_DCacheInvalidate();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = (int*)src0_addr0 + len_ch0;
    int *dst_addr1 = (int*)dst_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint32_t)src0_addr0);
    ALU_DST_ADDR_CH0_SET((uint32_t)dst_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint32_t)src0_addr1);
    ALU_DST_ADDR_CH1_SET((uint32_t)dst_addr1);

    ALU_MATHFUNC_CTRL_SET(1,1,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,0,op);

    wait_alu_mat_done();
    //Xil_DCacheInvalidate();
    ALU_RESET;
}

void hw_math::math_forward(void* src0_addr0, int len) {
    ALU_RESET;
    //Xil_DCacheInvalidate();

    int len_ch0 = (len / 2 + 7) & -8;
    int len_ch1 = u_align(len - len_ch0, 4);
    int *src0_addr1 = (int*)src0_addr0 + len_ch0;

    ALU_MATH_ALPHA_SET(*((uint32_t *)&alpha));
    ALU_MATH_BETA_SET(*((uint32_t *)&beta));

    ALU_VECLEN_CH0_SET(len_ch0);
    ALU_SRC0_ADDR_CH0_SET((uint32_t)src0_addr0);

    ALU_VECLEN_CH1_SET(len_ch1);
    ALU_SRC0_ADDR_CH1_SET((uint32_t)src0_addr1);

    ALU_MATHFUNC_CTRL_SET(1,1,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,0,op);

    wait_alu_mat_done();
    //Xil_DCacheInvalidate();
    ALU_RESET;
}

} // namespace accel
