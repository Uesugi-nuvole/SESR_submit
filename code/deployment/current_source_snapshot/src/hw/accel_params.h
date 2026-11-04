#ifndef __ACCEL_PARAMS_H_
#define __ACCEL_PARAMS_H_

#include <stdint.h>

#define u_align(x,n)  (((x) + (n) - 1) & -((n)))

#ifndef NCNN_TOOLS
#include "xil_io.h"

#define SESR_ACCEL_RECORDING() false
#define SESR_ACCEL_RECORD32(reg, value) ((void)0)
#define SESR_ACCEL_RECORD_ADDR(reg, value) ((void)0)

#define SESR_PACK_WH(w, h) ((((uint32_t)(w)) << 16) | ((uint32_t)(h) & 0xffffu))

#define SESR_SET_PORT32(reg, addr, value) \
    Xil_Out32((addr), (uint32_t)(value))

#define SESR_SET_PORT_ADDR(reg, addr, value) \
    Xil_Out32((addr), (uint32_t)(value))

#define ACCEL_REG_BASEADDR 0x00010000

#define ACCEL_REG_MATHFUNC_CTRL        (ACCEL_REG_BASEADDR + 0*4)
#define ACCEL_REG_ACTFUNC_CTRL         (ACCEL_REG_BASEADDR + 1*4)
#define ACCEL_REG_ALU_VECLEN0          (ACCEL_REG_BASEADDR + 2*4)
#define ACCEL_REG_SRC0_ADDR0           (ACCEL_REG_BASEADDR + 3*4)
#define ACCEL_REG_SRC1_ADDR0           (ACCEL_REG_BASEADDR + 4*4)
#define ACCEL_REG_DST_ADDR0            (ACCEL_REG_BASEADDR + 5*4)
#define ACCEL_REG_ALU_VECLEN1          (ACCEL_REG_BASEADDR + 6*4)
#define ACCEL_REG_SRC0_ADDR1           (ACCEL_REG_BASEADDR + 7*4)
#define ACCEL_REG_SRC1_ADDR1           (ACCEL_REG_BASEADDR + 8*4)
#define ACCEL_REG_DST_ADDR1            (ACCEL_REG_BASEADDR + 9*4)
#define ACCEL_REG_MATH_ALPHA           (ACCEL_REG_BASEADDR + 10*4)
#define ACCEL_REG_MATH_BETA            (ACCEL_REG_BASEADDR + 11*4)
#define ACCEL_REG_ACT_RANG_0           (ACCEL_REG_BASEADDR + 12*4)
#define ACCEL_REG_ACT_RANG_1           (ACCEL_REG_BASEADDR + 13*4)
#define ACCEL_REG_ACT_RANG_2           (ACCEL_REG_BASEADDR + 14*4)
#define ACCEL_REG_ACT_RANG_3           (ACCEL_REG_BASEADDR + 15*4)
#define ACCEL_REG_ACT_COEFF_A0         (ACCEL_REG_BASEADDR + 16*4)
#define ACCEL_REG_ACT_COEFF_A1         (ACCEL_REG_BASEADDR + 17*4)
#define ACCEL_REG_ACT_COEFF_A2         (ACCEL_REG_BASEADDR + 18*4)
#define ACCEL_REG_ACT_COEFF_A3         (ACCEL_REG_BASEADDR + 19*4)
#define ACCEL_REG_ACT_COEFF_A4         (ACCEL_REG_BASEADDR + 20*4)
#define ACCEL_REG_ACT_COEFF_B0         (ACCEL_REG_BASEADDR + 21*4)
#define ACCEL_REG_ACT_COEFF_B1         (ACCEL_REG_BASEADDR + 22*4)
#define ACCEL_REG_ACT_COEFF_B2         (ACCEL_REG_BASEADDR + 23*4)
#define ACCEL_REG_ACT_COEFF_B3         (ACCEL_REG_BASEADDR + 24*4)
#define ACCEL_REG_ACT_COEFF_B4         (ACCEL_REG_BASEADDR + 25*4)
#define ACCEL_REG_ACT_COEFF_C0         (ACCEL_REG_BASEADDR + 26*4)
#define ACCEL_REG_ACT_COEFF_C1         (ACCEL_REG_BASEADDR + 27*4)
#define ACCEL_REG_ACT_COEFF_C2         (ACCEL_REG_BASEADDR + 28*4)
#define ACCEL_REG_ACT_COEFF_C3         (ACCEL_REG_BASEADDR + 29*4)
#define ACCEL_REG_ACT_COEFF_C4         (ACCEL_REG_BASEADDR + 30*4)
#define ACCEL_REG_INNER_CTRL           (ACCEL_REG_BASEADDR + 31*4)
#define ACCEL_REG_ALU_ODATA            (ACCEL_REG_BASEADDR + 34*4)

#define ACCEL_REG_GEMM_CTRL            (ACCEL_REG_BASEADDR + 60*4)
#define ACCEL_REG_QUANT_DATA           (ACCEL_REG_BASEADDR + 61*4)
#define ACCEL_REG_REQUANT_DATA         (ACCEL_REG_BASEADDR + 62*4)
#define ACCEL_REG_DEQUANT_ADDR         (ACCEL_REG_BASEADDR + 63*4)
#define ACCEL_REG_BIAS_ADDR            (ACCEL_REG_BASEADDR + 64*4)
#define ACCEL_REG_SHAPE_IFM_C          (ACCEL_REG_BASEADDR + 65*4)
#define ACCEL_REG_SHAPE_IFM_WH         (ACCEL_REG_BASEADDR + 66*4)
#define ACCEL_REG_SHAPE_IFM_CSTEP      (ACCEL_REG_BASEADDR + 67*4)
#define ACCEL_REG_SHAPE_OFM_C          (ACCEL_REG_BASEADDR + 68*4)
#define ACCEL_REG_SHAPE_OFM_WH         (ACCEL_REG_BASEADDR + 69*4)
#define ACCEL_REG_SHAPE_OFM_CSTEP      (ACCEL_REG_BASEADDR + 70*4)
#define ACCEL_REG_WGT_LEN              (ACCEL_REG_BASEADDR + 71*4)
#define ACCEL_REG_IFM_BASEADDR         (ACCEL_REG_BASEADDR + 72*4)
#define ACCEL_REG_WGT_BASEADDR         (ACCEL_REG_BASEADDR + 73*4)
#define ACCEL_REG_OFM_BASEADDR         (ACCEL_REG_BASEADDR + 74*4)
#define ACCEL_REG_DIV_IFM_C            (ACCEL_REG_BASEADDR + 75*4)
#define ACCEL_REG_SOFT_RESET           (ACCEL_REG_BASEADDR + 76*4)
#define ACCEL_REG_QUANT_OFFSET         (ACCEL_REG_BASEADDR + 77*4)

#define ACCEL_REG_PIXELSHUFFLE_CTRL    (ACCEL_REG_BASEADDR + 80*4)
#define ACCEL_REG_PIXELSHUFFLE_IC      (ACCEL_REG_BASEADDR + 81*4)
#define ACCEL_REG_PIXELSHUFFLE_IWH     (ACCEL_REG_BASEADDR + 82*4)
#define ACCEL_REG_PIXELSHUFFLE_ICSTEP  (ACCEL_REG_BASEADDR + 83*4)
#define ACCEL_REG_PIXELSHUFFLE_OC      (ACCEL_REG_BASEADDR + 84*4)
#define ACCEL_REG_PIXELSHUFFLE_OWH     (ACCEL_REG_BASEADDR + 85*4)
#define ACCEL_REG_PIXELSHUFFLE_OCSTEP  (ACCEL_REG_BASEADDR + 86*4)
#define ACCEL_REG_PIXELSHUFFLE_IFM     (ACCEL_REG_BASEADDR + 87*4)
#define ACCEL_REG_PIXELSHUFFLE_OFM     (ACCEL_REG_BASEADDR + 88*4)

#define ACCEL_REG_SESR_IMAGE_CTRL      (ACCEL_REG_BASEADDR + 90*4)
#define ACCEL_REG_SESR_IMAGE_SHAPE     (ACCEL_REG_BASEADDR + 91*4)
#define ACCEL_REG_SESR_IMAGE_SCALE     (ACCEL_REG_BASEADDR + 92*4)
#define ACCEL_REG_SESR_IMAGE_SRC0_ADDR (ACCEL_REG_BASEADDR + 93*4)
#define ACCEL_REG_SESR_IMAGE_SRC1_ADDR (ACCEL_REG_BASEADDR + 94*4)
#define ACCEL_REG_SESR_IMAGE_SRC2_ADDR (ACCEL_REG_BASEADDR + 95*4)
#define ACCEL_REG_SESR_IMAGE_DST0_ADDR (ACCEL_REG_BASEADDR + 96*4)
#define ACCEL_REG_SESR_IMAGE_DST1_ADDR (ACCEL_REG_BASEADDR + 97*4)
#define ACCEL_REG_SESR_IMAGE_DST2_ADDR (ACCEL_REG_BASEADDR + 98*4)
#define ACCEL_REG_SESR_IMAGE_DST3_ADDR (ACCEL_REG_BASEADDR + 99*4)

#define ACCEL_REG_PERF_CYCLE_LO_ADDR  (ACCEL_REG_BASEADDR + 100*4)
#define ACCEL_REG_PERF_CYCLE_HI_ADDR  (ACCEL_REG_BASEADDR + 101*4)
#define ACCEL_REG_PERF_GEMM_LO_ADDR  (ACCEL_REG_BASEADDR + 102*4)
#define ACCEL_REG_PERF_GEMM_HI_ADDR  (ACCEL_REG_BASEADDR + 103*4)
#define ACCEL_REG_PERF_ALU_LO_ADDR  (ACCEL_REG_BASEADDR + 104*4)
#define ACCEL_REG_PERF_ALU_HI_ADDR  (ACCEL_REG_BASEADDR + 105*4)
#define ACCEL_REG_PERF_PIXEL_LO_ADDR  (ACCEL_REG_BASEADDR + 106*4)
#define ACCEL_REG_PERF_PIXEL_HI_ADDR  (ACCEL_REG_BASEADDR + 107*4)
#define ACCEL_REG_PERF_IMAGE_LO_ADDR  (ACCEL_REG_BASEADDR + 108*4)
#define ACCEL_REG_PERF_IMAGE_HI_ADDR  (ACCEL_REG_BASEADDR + 109*4)

#define ACCEL_REG_PERF_IMAGE_AXI_WAIT_LO_ADDR  (ACCEL_REG_BASEADDR + 114*4)
#define ACCEL_REG_PERF_IMAGE_AXI_WAIT_HI_ADDR  (ACCEL_REG_BASEADDR + 115*4)

/*****************************ALU*******************************/
#define ALU_RESET \
    do { \
        Xil_Out32(ACCEL_REG_MATHFUNC_CTRL, 0); \
        Xil_Out32(ACCEL_REG_ACTFUNC_CTRL, 0); \
        Xil_Out32(ACCEL_REG_INNER_CTRL, 0); \
    } while (0)

#define ALU_MATH_RESET \
        SESR_SET_PORT32(0, ACCEL_REG_MATHFUNC_CTRL, 0)

#define ALU_ACT_RESET \
        SESR_SET_PORT32(1, ACCEL_REG_ACTFUNC_CTRL, 0)

#define ALU_INNERPROD_RESET \
        SESR_SET_PORT32(31, ACCEL_REG_INNER_CTRL, 0)

#define ALU_MATHFUNC_CTRL_SET(en,src_num,mul_src1_sel,add_src0_sel,add_src1_sel,sub_en,add_en,mul_en,max_en,min_en,src1_broadcast,op) \
        SESR_SET_PORT32(0, ACCEL_REG_MATHFUNC_CTRL, \
        ((uint32_t)(en) | ((uint32_t)(src_num) << 1) | ((uint32_t)(mul_src1_sel) << 3) | ((uint32_t)(add_src0_sel) << 6) | ((uint32_t)(add_src1_sel) << 9) | \
        ((uint32_t)(sub_en) << 12) | ((uint32_t)(add_en) << 13) | ((uint32_t)(mul_en) << 14) | ((uint32_t)(max_en) << 15) | ((uint32_t)(min_en) << 16) | \
        ((uint32_t)(src1_broadcast) << 17) | ((uint32_t)(op) << 23)))

#define ALU_ACTFUNC_CTRL_SET(en,prop,src_sel,dst_sel,op) \
        SESR_SET_PORT32(1, ACCEL_REG_ACTFUNC_CTRL, \
        ((uint32_t)(en) | ((uint32_t)(prop) << 1) | ((uint32_t)(src_sel) << 3) | ((uint32_t)(dst_sel) << 5) | ((uint32_t)(op) << 16)))

#define ALU_INNERPROD_CTRL_SET(en) \
        SESR_SET_PORT32(31, ACCEL_REG_INNER_CTRL, (en))

#define ALU_VECLEN_CH0_SET(x) \
        SESR_SET_PORT32(2, ACCEL_REG_ALU_VECLEN0, (x))
#define ALU_SRC0_ADDR_CH0_SET(x) \
        SESR_SET_PORT_ADDR(3, ACCEL_REG_SRC0_ADDR0, (x))
#define ALU_SRC1_ADDR_CH0_SET(x) \
        SESR_SET_PORT_ADDR(4, ACCEL_REG_SRC1_ADDR0, (x))
#define ALU_DST_ADDR_CH0_SET(x) \
        SESR_SET_PORT_ADDR(5, ACCEL_REG_DST_ADDR0, (x))

#define ALU_VECLEN_CH1_SET(x) \
        SESR_SET_PORT32(6, ACCEL_REG_ALU_VECLEN1, (x))
#define ALU_SRC0_ADDR_CH1_SET(x) \
        SESR_SET_PORT_ADDR(7, ACCEL_REG_SRC0_ADDR1, (x))
#define ALU_SRC1_ADDR_CH1_SET(x) \
        SESR_SET_PORT_ADDR(8, ACCEL_REG_SRC1_ADDR1, (x))
#define ALU_DST_ADDR_CH1_SET(x) \
        SESR_SET_PORT_ADDR(9, ACCEL_REG_DST_ADDR1, (x))

#define ALU_MATH_ALPHA_SET(x) \
        SESR_SET_PORT32(10, ACCEL_REG_MATH_ALPHA, (x))
#define ALU_MATH_BETA_SET(x) \
        SESR_SET_PORT32(11, ACCEL_REG_MATH_BETA, (x))

#define ALU_ACT_COEFF_A_0_SET(x) \
        SESR_SET_PORT32(16, ACCEL_REG_ACT_COEFF_A0, (x))
#define ALU_ACT_COEFF_A_1_SET(x) \
        SESR_SET_PORT32(17, ACCEL_REG_ACT_COEFF_A1, (x))
#define ALU_ACT_COEFF_A_2_SET(x) \
        SESR_SET_PORT32(18, ACCEL_REG_ACT_COEFF_A2, (x))
#define ALU_ACT_COEFF_A_3_SET(x) \
        SESR_SET_PORT32(19, ACCEL_REG_ACT_COEFF_A3, (x))
#define ALU_ACT_COEFF_A_4_SET(x) \
        SESR_SET_PORT32(20, ACCEL_REG_ACT_COEFF_A4, (x))

#define ALU_ACT_COEFF_B_0_SET(x) \
        SESR_SET_PORT32(21, ACCEL_REG_ACT_COEFF_B0, (x))
#define ALU_ACT_COEFF_B_1_SET(x) \
        SESR_SET_PORT32(22, ACCEL_REG_ACT_COEFF_B1, (x))
#define ALU_ACT_COEFF_B_2_SET(x) \
        SESR_SET_PORT32(23, ACCEL_REG_ACT_COEFF_B2, (x))
#define ALU_ACT_COEFF_B_3_SET(x) \
        SESR_SET_PORT32(24, ACCEL_REG_ACT_COEFF_B3, (x))
#define ALU_ACT_COEFF_B_4_SET(x) \
        SESR_SET_PORT32(25, ACCEL_REG_ACT_COEFF_B4, (x))

#define ALU_ACT_COEFF_C_0_SET(x) \
        SESR_SET_PORT32(26, ACCEL_REG_ACT_COEFF_C0, (x))
#define ALU_ACT_COEFF_C_1_SET(x) \
        SESR_SET_PORT32(27, ACCEL_REG_ACT_COEFF_C1, (x))
#define ALU_ACT_COEFF_C_2_SET(x) \
        SESR_SET_PORT32(28, ACCEL_REG_ACT_COEFF_C2, (x))
#define ALU_ACT_COEFF_C_3_SET(x) \
        SESR_SET_PORT32(29, ACCEL_REG_ACT_COEFF_C3, (x))
#define ALU_ACT_COEFF_C_4_SET(x) \
        SESR_SET_PORT32(30, ACCEL_REG_ACT_COEFF_C4, (x))

#define ALU_ACT_RANGE_0_SET(x) \
        SESR_SET_PORT32(12, ACCEL_REG_ACT_RANG_0, (x))
#define ALU_ACT_RANGE_1_SET(x) \
        SESR_SET_PORT32(13, ACCEL_REG_ACT_RANG_1, (x))
#define ALU_ACT_RANGE_2_SET(x) \
        SESR_SET_PORT32(14, ACCEL_REG_ACT_RANG_2, (x))
#define ALU_ACT_RANGE_3_SET(x) \
        SESR_SET_PORT32(15, ACCEL_REG_ACT_RANG_3, (x))

/*****************************GEMM*******************************/
#define GEMM_RESET \
        SESR_SET_PORT32(60, ACCEL_REG_GEMM_CTRL, 0)

#define GEMM_CTRL_SET(en,op,kernel,stride,pad_mode,pad_left,pad_right,pad_top,pad_bottom,bias_en,requant_en,layout_en,oscale_en,div_ifm_c_en,no_writeback) \
        SESR_SET_PORT32(60, ACCEL_REG_GEMM_CTRL, \
        ((uint32_t)(en) | ((uint32_t)(op) << 1) | ((uint32_t)(kernel) << 3) | ((uint32_t)(stride) << 6) | ((uint32_t)(pad_mode) << 9) | \
        ((uint32_t)(pad_left) << 11) | ((uint32_t)(pad_right) << 13) | ((uint32_t)(pad_top) << 15) | ((uint32_t)(pad_bottom) << 17) | \
        ((uint32_t)(bias_en) << 19) | ((uint32_t)(requant_en) << 20) | ((uint32_t)(layout_en) << 21) | ((uint32_t)(oscale_en) << 22) | \
        ((uint32_t)(div_ifm_c_en) << 23) | ((uint32_t)(no_writeback) << 24)))
#define GEMM_CTRL_SET_EX(en,op,kernel,stride,pad_mode,pad_left,pad_right,pad_top,pad_bottom,bias_en,requant_en,layout_en,oscale_en,div_ifm_c_en,no_writeback,weight_cache_en) \
        SESR_SET_PORT32(60, ACCEL_REG_GEMM_CTRL, \
        ((uint32_t)(en) | ((uint32_t)(op) << 1) | ((uint32_t)(kernel) << 3) | ((uint32_t)(stride) << 6) | ((uint32_t)(pad_mode) << 9) | \
        ((uint32_t)(pad_left) << 11) | ((uint32_t)(pad_right) << 13) | ((uint32_t)(pad_top) << 15) | ((uint32_t)(pad_bottom) << 17) | \
        ((uint32_t)(bias_en) << 19) | ((uint32_t)(requant_en) << 20) | ((uint32_t)(layout_en) << 21) | ((uint32_t)(oscale_en) << 22) | \
        ((uint32_t)(div_ifm_c_en) << 23) | ((uint32_t)(no_writeback) << 24) | ((uint32_t)(weight_cache_en) << 25)))
#define GEMM_QUANT_DATA_SET(x) \
        SESR_SET_PORT32(61, ACCEL_REG_QUANT_DATA, (x))
#define GEMM_QUANT_OFFSET_SET(x) \
        SESR_SET_PORT32(77, ACCEL_REG_QUANT_OFFSET, (x))
#define GEMM_REQUANT_DATA_SET(x) \
        SESR_SET_PORT32(62, ACCEL_REG_REQUANT_DATA, (x))
#define GEMM_DEQUANT_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(63, ACCEL_REG_DEQUANT_ADDR, (x))
#define GEMM_BIAS_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(64, ACCEL_REG_BIAS_ADDR, (x))

#define GEMM_IFM_C_SET(c) \
        SESR_SET_PORT32(65, ACCEL_REG_SHAPE_IFM_C, (c))
#define GEMM_IFM_WH_SET(w,h) \
        SESR_SET_PORT32(66, ACCEL_REG_SHAPE_IFM_WH, SESR_PACK_WH((w), (h)))
#define GEMM_IFM_CSTEP_SET(x) \
        SESR_SET_PORT32(67, ACCEL_REG_SHAPE_IFM_CSTEP, (x))

#define GEMM_OFM_C_SET(c) \
        SESR_SET_PORT32(68, ACCEL_REG_SHAPE_OFM_C, (c))
#define GEMM_OFM_WH_SET(w,h) \
        SESR_SET_PORT32(69, ACCEL_REG_SHAPE_OFM_WH, SESR_PACK_WH((w), (h)))
#define GEMM_OFM_CSTEP_SET(x) \
        SESR_SET_PORT32(70, ACCEL_REG_SHAPE_OFM_CSTEP, (x))

#define GEMM_WGT_LEN_SET(x) \
        SESR_SET_PORT32(71, ACCEL_REG_WGT_LEN, (x))
#define GEMM_IFM_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(72, ACCEL_REG_IFM_BASEADDR, (x))
#define GEMM_WGT_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(73, ACCEL_REG_WGT_BASEADDR, (x))
#define GEMM_OFM_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(74, ACCEL_REG_OFM_BASEADDR, (x))
#define GEMM_DIV_IFM_C_SET(x) \
        SESR_SET_PORT32(75, ACCEL_REG_DIV_IFM_C, (x))

/*****************************PIXELSHUFFLE*******************************/
#define PIXELSHUFFLE_RESET \
        SESR_SET_PORT32(80, ACCEL_REG_PIXELSHUFFLE_CTRL, 0)

#define PIXELSHUFFLE_CTRL_SET(en,factor,mode) \
        SESR_SET_PORT32(80, ACCEL_REG_PIXELSHUFFLE_CTRL, ((uint32_t)(en) | ((uint32_t)(factor) << 3) | ((uint32_t)(mode) << 6)))

#define PIXELSHUFFLE_IFM_C_SET(c) \
        SESR_SET_PORT32(81, ACCEL_REG_PIXELSHUFFLE_IC, (c))
#define PIXELSHUFFLE_IFM_WH_SET(w,h) \
        SESR_SET_PORT32(82, ACCEL_REG_PIXELSHUFFLE_IWH, SESR_PACK_WH((w), (h)))
#define PIXELSHUFFLE_IFM_CSTEP_SET(x) \
        SESR_SET_PORT32(83, ACCEL_REG_PIXELSHUFFLE_ICSTEP, (x))

#define PIXELSHUFFLE_OFM_C_SET(c) \
        SESR_SET_PORT32(84, ACCEL_REG_PIXELSHUFFLE_OC, (c))
#define PIXELSHUFFLE_OFM_WH_SET(w,h) \
        SESR_SET_PORT32(85, ACCEL_REG_PIXELSHUFFLE_OWH, SESR_PACK_WH((w), (h)))
#define PIXELSHUFFLE_OFM_CSTEP_SET(x) \
        SESR_SET_PORT32(86, ACCEL_REG_PIXELSHUFFLE_OCSTEP, (x))

#define PIXELSHUFFLE_IFM_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(87, ACCEL_REG_PIXELSHUFFLE_IFM, (x))
#define PIXELSHUFFLE_OFM_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(88, ACCEL_REG_PIXELSHUFFLE_OFM, (x))

/*****************************SESR IMAGE PROC*******************************/
#define SESR_IMAGE_OP_PREPROCESS              0
#define SESR_IMAGE_OP_POSTPROCESS             1
#define SESR_IMAGE_OP_SPLIT                   2
#define SESR_IMAGE_OP_TAIL_FUSED              3
#define SESR_IMAGE_OP_TAIL_ADD_FUSED          4
#define SESR_IMAGE_OP_TAIL_ADD_STREAM_FUSED   5

#define SESR_IMAGE_RESET \
        SESR_SET_PORT32(90, ACCEL_REG_SESR_IMAGE_CTRL, 0)

#define SESR_IMAGE_CTRL_SET(en,op) \
        SESR_SET_PORT32(90, ACCEL_REG_SESR_IMAGE_CTRL, ((uint32_t)(en) | ((uint32_t)(op) << 1)))

#define SESR_IMAGE_SHAPE_SET(w,h) \
        SESR_SET_PORT32(91, ACCEL_REG_SESR_IMAGE_SHAPE, SESR_PACK_WH((w), (h)))

#define SESR_IMAGE_SCALE_SET(s) \
        SESR_SET_PORT32(92, ACCEL_REG_SESR_IMAGE_SCALE, (s))

#define SESR_IMAGE_SRC0_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(93, ACCEL_REG_SESR_IMAGE_SRC0_ADDR, (x))
#define SESR_IMAGE_SRC1_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(94, ACCEL_REG_SESR_IMAGE_SRC1_ADDR, (x))
#define SESR_IMAGE_SRC2_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(95, ACCEL_REG_SESR_IMAGE_SRC2_ADDR, (x))

#define SESR_IMAGE_DST0_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(96, ACCEL_REG_SESR_IMAGE_DST0_ADDR, (x))
#define SESR_IMAGE_DST1_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(97, ACCEL_REG_SESR_IMAGE_DST1_ADDR, (x))
#define SESR_IMAGE_DST2_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(98, ACCEL_REG_SESR_IMAGE_DST2_ADDR, (x))
#define SESR_IMAGE_DST3_ADDR_SET(x) \
        SESR_SET_PORT_ADDR(99, ACCEL_REG_SESR_IMAGE_DST3_ADDR, (x))

/*****************************CYCLE_COUNT*******************************/

#define TOTAL_CYCLES \
	Xil_In32(ACCEL_REG_PERF_CYCLE_LO_ADDR)
#define GEMM_CYCLES \
	Xil_In32(ACCEL_REG_PERF_GEMM_LO_ADDR)
#define ALU_CYCLES \
	Xil_In32(ACCEL_REG_PERF_ALU_LO_ADDR)
#define PIXELSHUFFLE_CYCLES \
	Xil_In32(ACCEL_REG_PERF_PIXEL_LO_ADDR)
#define IMAGE_CYCLES \
	Xil_In32(ACCEL_REG_PERF_IMAGE_LO_ADDR)
#define IMAGE_AXI_WAIT_CYCLES \
	Xil_In32(ACCEL_REG_PERF_IMAGE_AXI_WAIT_LO_ADDR)



/*****************************SOFT RESET*******************************/
#define SOFT_RESET \
        SESR_SET_PORT32(76, ACCEL_REG_SOFT_RESET, 1)

#define SOFT_RESET_CTRL_SET \
        SESR_SET_PORT32(76, ACCEL_REG_SOFT_RESET, 0)



#endif

#endif
