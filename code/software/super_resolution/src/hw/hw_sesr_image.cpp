#include "hw_sesr_image.h"

#include "../config.h"
#include "../utils/trace.h"
#include "accel_params.h"
#include "hw_command_sequencer.h"
#include <stdint.h>
#include <stdio.h>
#include "accel.h"
namespace accel {

namespace {

static bool aligned16(const void* p)
{
    return (((uintptr_t)p) & 15u) == 0u;
}

static int check_block16_pixels(int pixels, const char* name)
{
    if (pixels <= 0 || (pixels & 15) != 0)
    {
        printf("[ERR][sesr_image_hw] %s pixels=%d is not a positive multiple of 16\n", name, pixels);
        return -1;
    }
    return 0;
}

static int check_u8_mat(const cv::Mat& mat, int type, const char* name)
{
    if (mat.empty() || mat.type() != type || !aligned16(mat.data))
    {
        printf("[ERR][sesr_image_hw] invalid %s empty=%d type=%d addr=%p\n",
               name, mat.empty() ? 1 : 0, mat.type(), mat.data);
        return -1;
    }
    return 0;
}



struct TailAddStreamContext
{
    bool prepared;
    bool armed;
    int width;
    int height;
    int scale;
    const ncnn::Mat* skip_y;
    const cv::Mat* cb_lr;
    const cv::Mat* cr_lr;
    cv::Mat* bgr;
};

static TailAddStreamContext g_tail_stream = {};

} // namespace

int hw_sesr_preprocess_bgr_to_ycbcr_yinput(const cv::Mat& bgr, cv::Mat& y, cv::Mat& cb, cv::Mat& cr,
                                           ncnn::Mat& y_input, bool save_y)
{
    if (check_u8_mat(bgr, CV_8UC3, "bgr") != 0)
    {
        return -1;
    }

    const int pixels = bgr.rows * bgr.cols;
    if (check_block16_pixels(pixels, "preprocess") != 0)
    {
        return -1;
    }

    if (save_y)
    {
        y.create(bgr.rows, bgr.cols, CV_8UC1);
    }
    else
    {
        y.release();
    }
    cb.create(bgr.rows, bgr.cols, CV_8UC1);
    cr.create(bgr.rows, bgr.cols, CV_8UC1);
    y_input.create(bgr.cols, bgr.rows, 1, (size_t)4u);
    if ((save_y && (y.empty() || !aligned16(y.data))) ||
        cb.empty() || cr.empty() || y_input.empty() ||
        !aligned16(cb.data) || !aligned16(cr.data) || !aligned16(y_input.data))
    {
        printf("[ERR][sesr_image_hw] preprocess output allocation failed or unaligned\n");
        return -1;
    }

    SESR_IMAGE_RESET;
    //Xil_DCacheFlush();
    SESR_IMAGE_SHAPE_SET(bgr.cols, bgr.rows);
    SESR_IMAGE_SCALE_SET(1);
    SESR_IMAGE_SRC0_ADDR_SET((uint32_t)bgr.data);
    SESR_IMAGE_SRC1_ADDR_SET(0);
    SESR_IMAGE_SRC2_ADDR_SET(0);
    SESR_IMAGE_DST0_ADDR_SET(save_y ? (uint32_t)y.data : 0);
    SESR_IMAGE_DST1_ADDR_SET((uint32_t)cb.data);
    SESR_IMAGE_DST2_ADDR_SET((uint32_t)cr.data);
    SESR_IMAGE_DST3_ADDR_SET((uint32_t)y_input.data);
    SESR_IMAGE_CTRL_SET(1, SESR_IMAGE_OP_PREPROCESS);
    wait_image_done();
    //Xil_DCacheInvalidate();
    SESR_IMAGE_RESET;


    return 0;
}

int hw_sesr_postprocess_ycbcr_to_bgr(const ncnn::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    if (scale != 2 && scale != 4)
    {
        printf("[ERR][sesr_image_hw] unsupported postprocess scale=%d\n", scale);
        return -1;
    }

    if (y_sr.empty() || y_sr.dims != 3 || y_sr.c != 1 || y_sr.elemsize != (size_t)4u || !aligned16(y_sr.data))
    {
        printf("[ERR][sesr_image_hw] invalid y_sr dims=%d w=%d h=%d c=%d elemsize=%zu addr=%p\n",
               y_sr.dims, y_sr.w, y_sr.h, y_sr.c, y_sr.elemsize, y_sr.data);
        return -1;
    }
    if (check_u8_mat(cb_lr, CV_8UC1, "cb_lr") != 0 || check_u8_mat(cr_lr, CV_8UC1, "cr_lr") != 0)
    {
        return -1;
    }
    if (cb_lr.rows != cr_lr.rows || cb_lr.cols != cr_lr.cols ||
        y_sr.h != cb_lr.rows * scale || y_sr.w != cb_lr.cols * scale)
    {
        printf("[ERR][sesr_image_hw] postprocess shape mismatch y=%dx%d cb=%dx%d cr=%dx%d scale=%d\n",
               y_sr.w, y_sr.h, cb_lr.cols, cb_lr.rows, cr_lr.cols, cr_lr.rows, scale);
        return -1;
    }

    const int out_pixels = y_sr.w * y_sr.h;
    if (check_block16_pixels(out_pixels, "postprocess") != 0)
    {
        return -1;
    }

    bgr.create(y_sr.h, y_sr.w, CV_8UC3);
    if (bgr.empty() || !aligned16(bgr.data))
    {
        printf("[ERR][sesr_image_hw] postprocess bgr allocation failed or unaligned\n");
        return -1;
    }

    SESR_IMAGE_RESET;
    //Xil_DCacheFlush();
    SESR_IMAGE_SHAPE_SET(cb_lr.cols, cb_lr.rows);
    SESR_IMAGE_SCALE_SET(scale);
    SESR_IMAGE_SRC0_ADDR_SET((uint32_t)y_sr.data);
    SESR_IMAGE_SRC1_ADDR_SET((uint32_t)cb_lr.data);
    SESR_IMAGE_SRC2_ADDR_SET((uint32_t)cr_lr.data);
    SESR_IMAGE_DST0_ADDR_SET((uint32_t)bgr.data);
    SESR_IMAGE_DST1_ADDR_SET(0);
    SESR_IMAGE_DST2_ADDR_SET(0);
    SESR_IMAGE_DST3_ADDR_SET(0);
    SESR_IMAGE_CTRL_SET(1, SESR_IMAGE_OP_POSTPROCESS);
    wait_image_done();
    //Xil_DCacheInvalidate();
    SESR_IMAGE_RESET;


    return 0;
}

int hw_sesr_tail_fused_to_bgr(const ncnn::Mat& tail_blob, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{

    if (scale != 2 && scale != 4)
    {
        printf("[ERR][sesr_image_hw] unsupported fused tail scale=%d\n", scale);
        return -1;
    }

    const int expected_c = scale * scale;
    if (tail_blob.empty() || tail_blob.dims != 3 || tail_blob.c != expected_c ||
        tail_blob.elemsize != (size_t)4u || tail_blob.elempack != 1 || !aligned16(tail_blob.data))
    {
        printf("[ERR][sesr_image_hw] invalid fused tail dims=%d w=%d h=%d c=%d cstep=%zu elemsize=%zu elempack=%d addr=%p scale=%d\n",
               tail_blob.dims, tail_blob.w, tail_blob.h, tail_blob.c, tail_blob.cstep,
               tail_blob.elemsize, tail_blob.elempack, tail_blob.data, scale);
        return -1;
    }
    if (check_u8_mat(cb_lr, CV_8UC1, "cb_lr") != 0 || check_u8_mat(cr_lr, CV_8UC1, "cr_lr") != 0)
    {
        return -1;
    }
    if (cb_lr.rows != cr_lr.rows || cb_lr.cols != cr_lr.cols ||
        tail_blob.h != cb_lr.rows || tail_blob.w != cb_lr.cols)
    {
        printf("[ERR][sesr_image_hw] fused tail shape mismatch tail=%dx%dx%d cb=%dx%d cr=%dx%d scale=%d\n",
               tail_blob.w, tail_blob.h, tail_blob.c, cb_lr.cols, cb_lr.rows, cr_lr.cols, cr_lr.rows, scale);
        return -1;
    }

    const int out_w = tail_blob.w * scale;
    const int out_h = tail_blob.h * scale;
    const int out_pixels = out_w * out_h;
    if (check_block16_pixels(out_pixels, "fused tail") != 0)
    {
        return -1;
    }

    bgr.create(out_h, out_w, CV_8UC3);
    if (bgr.empty() || !aligned16(bgr.data))
    {
        printf("[ERR][sesr_image_hw] fused tail bgr allocation failed or unaligned\n");
        return -1;
    }

    SESR_IMAGE_RESET;
    //Xil_DCacheFlush();
    SESR_IMAGE_SHAPE_SET(tail_blob.w, tail_blob.h);
    SESR_IMAGE_SCALE_SET(scale);
    SESR_IMAGE_SRC0_ADDR_SET((uint32_t)tail_blob.data);
    SESR_IMAGE_SRC1_ADDR_SET((uint32_t)cb_lr.data);
    SESR_IMAGE_SRC2_ADDR_SET((uint32_t)cr_lr.data);
    SESR_IMAGE_DST0_ADDR_SET((uint32_t)bgr.data);
    SESR_IMAGE_DST1_ADDR_SET(0);
    SESR_IMAGE_DST2_ADDR_SET(0);
    SESR_IMAGE_DST3_ADDR_SET(0);
    SESR_IMAGE_CTRL_SET(1, SESR_IMAGE_OP_TAIL_FUSED);
    wait_image_done();
    //Xil_DCacheInvalidate();
    SESR_IMAGE_RESET;


    return 0;
}

int hw_sesr_tail_add_fused_to_bgr(const ncnn::Mat& tail_blob, const ncnn::Mat& skip_y,
                                  const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    if (scale != 2 && scale != 4)
    {
        printf("[ERR][sesr_image_hw] unsupported fused tail-add scale=%d\n", scale);
        return -1;
    }

    const int expected_c = scale * scale;
    if (tail_blob.empty() || tail_blob.dims != 3 || tail_blob.c != expected_c ||
        tail_blob.elemsize != (size_t)4u || tail_blob.elempack != 1 || !aligned16(tail_blob.data))
    {
        printf("[ERR][sesr_image_hw] invalid fused tail-add tail dims=%d w=%d h=%d c=%d cstep=%zu elemsize=%zu elempack=%d addr=%p scale=%d\n",
               tail_blob.dims, tail_blob.w, tail_blob.h, tail_blob.c, tail_blob.cstep,
               tail_blob.elemsize, tail_blob.elempack, tail_blob.data, scale);
        return -1;
    }
    if (skip_y.empty() || skip_y.dims != 3 || skip_y.c != 1 ||
        skip_y.w != tail_blob.w || skip_y.h != tail_blob.h ||
        skip_y.elemsize != (size_t)4u || skip_y.elempack != 1 || !aligned16(skip_y.data))
    {
        printf("[ERR][sesr_image_hw] invalid fused tail-add skip dims=%d w=%d h=%d c=%d cstep=%zu elemsize=%zu elempack=%d addr=%p expected=%dx%d\n",
               skip_y.dims, skip_y.w, skip_y.h, skip_y.c, skip_y.cstep,
               skip_y.elemsize, skip_y.elempack, skip_y.data, tail_blob.w, tail_blob.h);
        return -1;
    }
    if (check_u8_mat(cb_lr, CV_8UC1, "cb_lr") != 0 || check_u8_mat(cr_lr, CV_8UC1, "cr_lr") != 0)
    {
        return -1;
    }
    if (cb_lr.rows != cr_lr.rows || cb_lr.cols != cr_lr.cols ||
        tail_blob.h != cb_lr.rows || tail_blob.w != cb_lr.cols)
    {
        printf("[ERR][sesr_image_hw] fused tail-add shape mismatch tail=%dx%dx%d skip=%dx%d cb=%dx%d cr=%dx%d scale=%d\n",
               tail_blob.w, tail_blob.h, tail_blob.c, skip_y.w, skip_y.h,
               cb_lr.cols, cb_lr.rows, cr_lr.cols, cr_lr.rows, scale);
        return -1;
    }

    const int out_w = tail_blob.w * scale;
    const int out_h = tail_blob.h * scale;
    const int out_pixels = out_w * out_h;
    if (check_block16_pixels(out_pixels, "fused tail-add") != 0)
    {
        return -1;
    }

    bgr.create(out_h, out_w, CV_8UC3);
    if (bgr.empty() || !aligned16(bgr.data))
    {
        printf("[ERR][sesr_image_hw] fused tail-add bgr allocation failed or unaligned\n");
        return -1;
    }

    SESR_IMAGE_RESET;
    //Xil_DCacheFlush();
    SESR_IMAGE_SHAPE_SET(tail_blob.w, tail_blob.h);
    SESR_IMAGE_SCALE_SET(scale);
    SESR_IMAGE_SRC0_ADDR_SET((uint32_t)tail_blob.data);
    SESR_IMAGE_SRC1_ADDR_SET((uint32_t)cb_lr.data);
    SESR_IMAGE_SRC2_ADDR_SET((uint32_t)cr_lr.data);
    SESR_IMAGE_DST0_ADDR_SET((uint32_t)bgr.data);
    SESR_IMAGE_DST1_ADDR_SET((uint32_t)skip_y.data);
    SESR_IMAGE_DST2_ADDR_SET(0);
    SESR_IMAGE_DST3_ADDR_SET(0);
    SESR_IMAGE_CTRL_SET(1, SESR_IMAGE_OP_TAIL_ADD_FUSED);
    wait_image_done();
    //Xil_DCacheInvalidate();
    SESR_IMAGE_RESET;


    return 0;
}

int hw_sesr_tail_add_stream_begin(int width, int height, const ncnn::Mat& skip_y,
                                  const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    if (scale != 2 && scale != 4)
    {
        printf("[ERR][sesr_image_hw] unsupported stream tail-add scale=%d\n", scale);
        return -1;
    }

    if (width <= 0 || height <= 0)
    {
        printf("[ERR][sesr_image_hw] invalid stream tail-add size=%dx%d\n", width, height);
        return -1;
    }

    if (skip_y.empty() || skip_y.dims != 3 || skip_y.c != 1 ||
        skip_y.w != width || skip_y.h != height ||
        skip_y.elemsize != (size_t)4u || skip_y.elempack != 1 || !aligned16(skip_y.data))
    {
        printf("[ERR][sesr_image_hw] invalid stream tail-add skip dims=%d w=%d h=%d c=%d cstep=%zu elemsize=%zu elempack=%d addr=%p expected=%dx%d\n",
               skip_y.dims, skip_y.w, skip_y.h, skip_y.c, skip_y.cstep,
               skip_y.elemsize, skip_y.elempack, skip_y.data, width, height);
        return -1;
    }

    if (check_u8_mat(cb_lr, CV_8UC1, "cb_lr") != 0 || check_u8_mat(cr_lr, CV_8UC1, "cr_lr") != 0)
    {
        return -1;
    }

    if (cb_lr.rows != height || cb_lr.cols != width ||
        cr_lr.rows != height || cr_lr.cols != width)
    {
        printf("[ERR][sesr_image_hw] stream tail-add shape mismatch skip=%dx%d cb=%dx%d cr=%dx%d scale=%d\n",
               width, height, cb_lr.cols, cb_lr.rows, cr_lr.cols, cr_lr.rows, scale);
        return -1;
    }

    const int out_w = width * scale;
    const int out_h = height * scale;
    const int out_pixels = out_w * out_h;
    if (check_block16_pixels(out_pixels, "stream tail-add") != 0)
    {
        return -1;
    }

    bgr.create(out_h, out_w, CV_8UC3);
    if (bgr.empty() || !aligned16(bgr.data))
    {
        printf("[ERR][sesr_image_hw] stream tail-add bgr allocation failed or unaligned\n");
        return -1;
    }

    SESR_IMAGE_RESET;
    ////Xil_DCacheInvalidate();
    SESR_IMAGE_SHAPE_SET(width, height);
    SESR_IMAGE_SCALE_SET(scale);
    SESR_IMAGE_SRC0_ADDR_SET(0);
    SESR_IMAGE_SRC1_ADDR_SET((uint32_t)cb_lr.data);
    SESR_IMAGE_SRC2_ADDR_SET((uint32_t)cr_lr.data);
    SESR_IMAGE_DST0_ADDR_SET((uint32_t)bgr.data);
    SESR_IMAGE_DST1_ADDR_SET((uint32_t)skip_y.data);
    SESR_IMAGE_DST2_ADDR_SET(0);
    SESR_IMAGE_DST3_ADDR_SET(0);
    SESR_IMAGE_CTRL_SET(1, SESR_IMAGE_OP_TAIL_ADD_STREAM_FUSED);



    return 0;
}

void hw_sesr_tail_add_stream_wait()
{
    if (!g_tail_stream.armed)
    {
        printf("[ERR][sesr_image_hw] stream tail-add was not armed before wait\n");
        hw_sesr_tail_add_stream_clear();
        return;
    }

    wait_image_done();
    Xil_DCacheInvalidate();
    SESR_IMAGE_RESET;

    hw_sesr_tail_add_stream_clear();
}

void hw_sesr_tail_add_stream_prepare(int width, int height, const ncnn::Mat& skip_y,
                                     const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    g_tail_stream.prepared = true;
    g_tail_stream.armed = false;
    g_tail_stream.width = width;
    g_tail_stream.height = height;
    g_tail_stream.scale = scale;
    g_tail_stream.skip_y = &skip_y;
    g_tail_stream.cb_lr = &cb_lr;
    g_tail_stream.cr_lr = &cr_lr;
    g_tail_stream.bgr = &bgr;
}

int hw_sesr_tail_add_stream_arm_if_match(int ofm_w, int ofm_h, int ofm_c)
{

    if (!g_tail_stream.prepared || g_tail_stream.armed)
    {
        return 0;
    }

    if (ofm_w != g_tail_stream.width || ofm_h != g_tail_stream.height ||
        ofm_c != g_tail_stream.scale * g_tail_stream.scale)
    {
        return 0;
    }

    if (hw_sesr_tail_add_stream_begin(g_tail_stream.width, g_tail_stream.height,
                                      *g_tail_stream.skip_y, *g_tail_stream.cb_lr, *g_tail_stream.cr_lr,
                                      g_tail_stream.scale, *g_tail_stream.bgr) != 0)
    {
        return -1;
    }

    g_tail_stream.armed = true;
    SESR_TRACE_STAGE("NPU", "tail-add-stream-armed", "input=%dx%d scale=x%d outc=%d",
                     ofm_w, ofm_h, g_tail_stream.scale, ofm_c);
    return 1;
}

void hw_sesr_tail_add_stream_clear()
{
    g_tail_stream = TailAddStreamContext();
}

int hw_sesr_split_ack(const ncnn::Mat& bottom_blob, size_t outputs)
{
    if (bottom_blob.empty())
    {
        return -1;
    }

    SESR_IMAGE_RESET;
    Xil_DCacheFlush();
    SESR_IMAGE_SHAPE_SET(bottom_blob.w, bottom_blob.h);
    SESR_IMAGE_SCALE_SET((uint32_t)outputs);
    SESR_IMAGE_SRC0_ADDR_SET((uint32_t)bottom_blob.data);
    SESR_IMAGE_SRC1_ADDR_SET(0);
    SESR_IMAGE_SRC2_ADDR_SET(0);
    SESR_IMAGE_DST0_ADDR_SET(0);
    SESR_IMAGE_DST1_ADDR_SET(0);
    SESR_IMAGE_DST2_ADDR_SET(0);
    SESR_IMAGE_DST3_ADDR_SET(0);
    SESR_IMAGE_CTRL_SET(1, SESR_IMAGE_OP_SPLIT);
    wait_image_done();
    Xil_DCacheInvalidate();
    SESR_IMAGE_RESET;


    return 0;
}

} // namespace accel
