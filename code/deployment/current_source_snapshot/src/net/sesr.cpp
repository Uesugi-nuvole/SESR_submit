#include "../ncnn/net.h"
#include "../ncnn/simpleocv.h"
#include "../layer/pixelshuffle.h"
#include "../hw/hw_math.h"
#include "../hw/hw_sesr_image.h"
#include "../hw/accel_params.h"
#include <stdio.h>
#include <algorithm>
#include <string.h>
#include "../config.h"
#include "../utils/trace.h"
#include "color_convert.h"
#include "sesr_m5_command_table.h"
#include "../net/timer.h"



#if SESR_SCALE==2
#include "../test_imgs/input_image_640x360.h"
#include "../models/sesr_m5_x2_qat.h"

#else
#include "../test_imgs/input_image_320x180.h"
#include "../models/sesr_m5_x4_qat.h"
#endif


namespace demo {

namespace {

static inline float clamp_255(float v)
{
    return std::min(std::max(v, 0.f), 255.f);
}

static inline unsigned char float_to_u8(float v)
{
    return (unsigned char)(clamp_255(v) + 0.5f);
}

static inline unsigned char float_unit_to_u8_bits(float v)
{
    uint32_t bits;
    memcpy(&bits, &v, sizeof(bits));

    const uint32_t sign = bits >> 31;
    const uint32_t exp = (bits >> 23) & 0xffu;
    const uint32_t mant = (1u << 23) | (bits & 0x7fffffu);
    if (sign || exp == 0 || exp < 119)
    {
        return 0;
    }
    if (exp >= 127)
    {
        return 255;
    }

    const int shift = 150 - (int)exp;
    const uint64_t prod = (uint64_t)mant * 255u;
    const uint64_t rounded = (prod + (1ull << (shift - 1))) >> shift;
    return rounded > 255u ? 255 : (unsigned char)rounded;
}

static inline unsigned char fixed_to_u8(int v)
{
    v = (v + (1 << 15)) >> 16;
    if (v < 0)
        return 0;
    if (v > 255)
        return 255;
    return (unsigned char)v;
}

static inline void ycbcr_fixed_to_bgr_store(int y_u8, int cb_u8, int cr_u8, uchar* bgr)
{
    const int yy = y_u8 << 16;
    const int cb = cb_u8 - 128;
    const int cr = cr_u8 - 128;

    bgr[0] = fixed_to_u8(yy + 116130 * cb);
    bgr[1] = fixed_to_u8(yy - 22554 * cb - 46802 * cr);
    bgr[2] = fixed_to_u8(yy + 91881 * cr);
}



int load_sesr_model(ncnn::Net& sesr)
{
    sesr.opt.use_int8_inference = true;

#ifdef SESR_FIXED_COMMAND_TABLE
    const ncnn::FixedParamTable* table = sesr_fixed_param_table(SESR_SCALE);
    if (!table)
    {
        printf("[ERR][sesr] fixed command table not found for scale x%d\n", SESR_SCALE);
        return -1;
    }
    if (sesr.load_fixed_param(*table))
    {
        printf("[ERR][sesr] load_fixed_param failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("CPU", "load-fixed-command-table", "table=%s layers=%d blobs=%d",
                     sesr_fixed_param_table_name(SESR_SCALE), table->layer_count, table->blob_count);
#else
#if SESR_SCALE==2
    if (sesr.load_param_mem(sesr_x2_qat_param))
    {
        printf("[ERR][sesr] load_param failed\n");
        return -1;
    }
#else
    if (sesr.load_param_mem(sesr_x4_qat_param))
    {
        printf("[ERR][sesr] load_param failed\n");
        return -1;
    }
#endif
#endif
#if SESR_SCALE==2
    if (sesr.load_model_mem((const char*)sesr_x2_qat_bin))
    {
        printf("[ERR][sesr] load_model failed\n");
        return -1;
    }
#else
    if (sesr.load_model_mem((const char*)sesr_x4_qat_bin))
    {
        printf("[ERR][sesr] load_model failed\n");
        return -1;
    }
#endif
    return 0;
}

int run_sesr_on_y_input(ncnn::Net& sesr, const ncnn::Mat& y_input, ncnn::Mat& y_sr)
{

    if (y_input.empty() || y_input.dims != 3 || y_input.c != 1 || y_input.elemsize != (size_t)4u)
    {
        printf("[ERR][sesr] invalid Y input tensor\n");
        return -1;
    }

#ifdef SESR_VERBOSE_LOG
    printf("[DBG][sesr] Y input tensor shape: w=%d h=%d c=1\n", y_input.w, y_input.h);
#endif

    ncnn::Extractor ex = sesr.create_extractor();
    int ret = ex.input("data", y_input);
    if (ret != 0)
    {
        printf("[ERR][sesr] ex.input(\"data\", y) failed, ret=%d\n", ret);
        return -1;
    }

#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ADD_ON_NPU)
    const char* output_blob_name = "output_conv_out";
#elif defined(SESR_FUSED_TAIL_ON_NPU)
    const char* output_blob_name = "res2_out";
#else
    const char* output_blob_name = "output";
#endif

    ret = ex.extract(output_blob_name, y_sr);
    if (ret != 0)
    {
        printf("[ERR][sesr] ex.extract(\"%s\", y_sr) failed, ret=%d\n", output_blob_name, ret);
        return -1;
    }
#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ON_NPU)
    if (y_sr.empty() || y_sr.c != SESR_SCALE * SESR_SCALE)
    {
        printf("[ERR][sesr] expected fused-tail blob c=%d, but got empty=%d c=%d\n",
               SESR_SCALE * SESR_SCALE, y_sr.empty() ? 1 : 0, y_sr.c);
        return -1;
    }
#else
    if (y_sr.empty() || y_sr.c != 1)
    {
        printf("[ERR][sesr] expected Y-only output, but got empty=%d c=%d\n", y_sr.empty() ? 1 : 0, y_sr.c);
        return -1;
    }
#endif


#ifdef SESR_VERBOSE_LOG
    printf("[DBG][sesr] extracted tensor %s shape: w=%d h=%d c=%d cstep=%zu\n",
           output_blob_name, y_sr.w, y_sr.h, y_sr.c, y_sr.cstep);
#endif
    return 0;
}

int run_sesr_fixed_command(ncnn::Net& sesr, const ncnn::FixedParamTable& table,
                           const ncnn::Mat& y_input, ncnn::Mat& y_sr)
{
    if (y_input.empty() || y_input.dims != 3 || y_input.c != 1 || y_input.elemsize != (size_t)4u)
    {
        printf("[ERR][sesr] invalid Y input tensor for fixed command\n");
        return -1;
    }
    if (table.input_blob < 0 || table.input_blob >= table.blob_count ||
        table.output_blob < 0 || table.output_blob >= table.blob_count ||
        table.fused_tail_blob < 0 || table.fused_tail_blob >= table.blob_count)
    {
        printf("[ERR][sesr] invalid fixed command blob indexes input=%d output=%d fused_tail=%d blobs=%d\n",
               table.input_blob, table.output_blob, table.fused_tail_blob, table.blob_count);
        return -1;
    }

    ncnn::Extractor ex = sesr.create_extractor();
    int ret = ex.input(table.input_blob, y_input);
    if (ret != 0)
    {
        printf("[ERR][sesr] fixed command input blob=%d failed, ret=%d\n", table.input_blob, ret);
        return -1;
    }

#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ON_NPU)
    const int output_blob_index = table.fused_tail_blob;
    const char* output_desc = "fused_tail";
#else
    const int output_blob_index = table.output_blob;
    const char* output_desc = "output";
#endif

    SESR_TRACE_STAGE("NPU", "e2e-fixed-command-start",
                     "table=%s input_blob=%d output_blob=%d output=%s",
                     sesr_fixed_param_table_name(SESR_SCALE), table.input_blob, output_blob_index, output_desc);
    ret = ex.extract(output_blob_index, y_sr);
    if (ret != 0)
    {
        printf("[ERR][sesr] fixed command extract blob=%d (%s) failed, ret=%d\n",
               output_blob_index, output_desc, ret);
        return -1;
    }
    SESR_TRACE_STAGE("NPU", "e2e-fixed-command-done",
                     "output=%s shape=%dx%dx%d", output_desc, y_sr.w, y_sr.h, y_sr.c);

#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ON_NPU)
    if (y_sr.empty() || y_sr.c != SESR_SCALE * SESR_SCALE)
    {
        printf("[ERR][sesr] fixed command expected fused-tail c=%d, but got empty=%d c=%d\n",
               SESR_SCALE * SESR_SCALE, y_sr.empty() ? 1 : 0, y_sr.c);
        return -1;
    }
#else
    if (y_sr.empty() || y_sr.c != 1)
    {
        printf("[ERR][sesr] fixed command expected Y-only output, but got empty=%d c=%d\n",
               y_sr.empty() ? 1 : 0, y_sr.c);
        return -1;
    }
#endif

    return 0;
}

int compose_bgr_from_tail_blob_cpu_ref(const ncnn::Mat& tail_blob, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    if (tail_blob.empty() || tail_blob.dims != 3 || tail_blob.c != scale * scale ||
        tail_blob.w != cb_lr.cols || tail_blob.h != cb_lr.rows ||
        cb_lr.empty() || cr_lr.empty() || cb_lr.rows != cr_lr.rows || cb_lr.cols != cr_lr.cols ||
        cb_lr.type() != CV_8UC1 || cr_lr.type() != CV_8UC1)
    {
        return -1;
    }

    const int out_w = tail_blob.w * scale;
    const int out_h = tail_blob.h * scale;
    bgr.create(out_h, out_w, CV_8UC3);
    if (bgr.empty())
    {
        return -1;
    }

    for (int oy = 0; oy < out_h; oy++)
    {
        uchar* outptr = bgr.ptr(oy);
        const int iy = oy / scale;
        const int ry = oy % scale;
        const uchar* cbrow = cb_lr.ptr(iy);
        const uchar* crrow = cr_lr.ptr(iy);
        for (int ox = 0; ox < out_w; ox++)
        {
            const int ix = ox / scale;
            const int rx = ox % scale;
            int q;
            if (scale == 4)
            {
                q = ((ry >> 1) << 3) | ((rx >> 1) << 2) | ((ry & 1) << 1) | (rx & 1);
            }
            else
            {
                q = (ry << 1) | rx;
            }
            const float* chptr = (const float*)tail_blob.channel(q);
            const unsigned char y_u8 = float_unit_to_u8_bits(chptr[(size_t)iy * tail_blob.w + ix]);
            ycbcr_fixed_to_bgr_store(y_u8, cbrow[ix], crrow[ix], outptr + ox * 3);
        }
    }

    return 0;
}

int compose_bgr_from_tail_add_blob_cpu_ref(const ncnn::Mat& tail_blob, const ncnn::Mat& skip_y,
                                           const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    if (tail_blob.empty() || tail_blob.dims != 3 || tail_blob.c != scale * scale ||
        skip_y.empty() || skip_y.dims != 3 || skip_y.c != 1 ||
        skip_y.w != tail_blob.w || skip_y.h != tail_blob.h ||
        tail_blob.w != cb_lr.cols || tail_blob.h != cb_lr.rows ||
        cb_lr.empty() || cr_lr.empty() || cb_lr.rows != cr_lr.rows || cb_lr.cols != cr_lr.cols ||
        cb_lr.type() != CV_8UC1 || cr_lr.type() != CV_8UC1)
    {
        return -1;
    }

    const int out_w = tail_blob.w * scale;
    const int out_h = tail_blob.h * scale;
    bgr.create(out_h, out_w, CV_8UC3);
    if (bgr.empty())
    {
        return -1;
    }

    const float* skip_ptr = (const float*)skip_y.channel(0);
    for (int oy = 0; oy < out_h; oy++)
    {
        uchar* outptr = bgr.ptr(oy);
        const int iy = oy / scale;
        const int ry = oy % scale;
        const uchar* cbrow = cb_lr.ptr(iy);
        const uchar* crrow = cr_lr.ptr(iy);
        for (int ox = 0; ox < out_w; ox++)
        {
            const int ix = ox / scale;
            const int rx = ox % scale;
            int q;
            if (scale == 4)
            {
                q = ((ry >> 1) << 3) | ((rx >> 1) << 2) | ((ry & 1) << 1) | (rx & 1);
            }
            else
            {
                q = (ry << 1) | rx;
            }
            const float* chptr = (const float*)tail_blob.channel(q);
            const float y_float = chptr[(size_t)iy * tail_blob.w + ix] + skip_ptr[(size_t)iy * skip_y.w + ix];
            const unsigned char y_u8 = float_unit_to_u8_bits(y_float);
            ycbcr_fixed_to_bgr_store(y_u8, cbrow[ix], crrow[ix], outptr + ox * 3);
        }
    }

    return 0;
}

int mat_u8_to_ncnn_plane(const cv::Mat& src, ncnn::Mat& dst)
{
    if (src.empty() || src.type() != CV_8UC1)
    {
        return -1;
    }

    dst.create(src.cols, src.rows, 1, (size_t)4u);
    if (dst.empty())
    {
        return -1;
    }

    float* dstptr = (float*)dst.channel(0);
    for (int y = 0; y < src.rows; y++)
    {
        const uchar* row = src.ptr(y);
        for (int x = 0; x < src.cols; x++)
        {
            dstptr[y * src.cols + x] = (float)row[x];
        }
    }

    return 0;
}

int replicate_plane_for_pixelshuffle(const ncnn::Mat& src_plane, ncnn::Mat& dst)
{
    if (src_plane.empty() || src_plane.dims != 3 || src_plane.c != 1)
    {
        return -1;
    }

    dst.create(src_plane.w, src_plane.h, 4, (size_t)4u);
    if (dst.empty())
    {
        return -1;
    }

    const float* srcptr = (const float*)src_plane.channel(0);
    const size_t plane_elems = (size_t)src_plane.w * src_plane.h;

    for (int q = 0; q < 4; q++)
    {
        float* dstptr = (float*)dst.channel(q);
        for (size_t i = 0; i < plane_elems; i++)
        {
            dstptr[i] = srcptr[i];
        }
    }

    return 0;
}

int pixelshuffle_x2(const ncnn::Mat& src, ncnn::Mat& dst)
{
#ifdef FORWARD_ON_NPU_PIXELSHUFFLE
    const bool pixelshuffle_npu =
        src.dims == 3 &&
        src.c % 4 == 0 &&
        src.elemsize == 4u &&
        src.elempack == 1 &&
        (src.w * 2) % 4 == 0;

    if (!pixelshuffle_npu)
    {
        printf("[ERR][sesr] chroma PixelShuffle cannot use NPU: w=%d h=%d c=%d dims=%d elemsize=%zu elempack=%d\n",
               src.w, src.h, src.c, src.dims, src.elemsize, src.elempack);
        return -1;
    }
#endif

    ncnn::PixelShuffle pixelshuffle;
    pixelshuffle.upscale_factor = 2;
    pixelshuffle.mode = 0;

    ncnn::Option opt;
    int ret = pixelshuffle.forward(src, dst, opt);
    if (ret != 0)
    {
        printf("[ERR][sesr] PixelShuffle x2 failed, ret=%d\n", ret);
        return -1;
    }

    return 0;
}

int ncnn_plane_to_mat_u8(const ncnn::Mat& src, cv::Mat& dst)
{
    if (src.empty() || src.dims != 3 || src.c != 1)
    {
        return -1;
    }

    dst.create(src.h, src.w, CV_8UC1);
    if (dst.empty())
    {
        return -1;
    }

    const float* srcptr = (const float*)src.channel(0);
    for (int y = 0; y < src.h; y++)
    {
        uchar* row = dst.ptr(y);
        for (int x = 0; x < src.w; x++)
        {
            row[x] = float_to_u8(srcptr[y * src.w + x]);
        }
    }

    return 0;
}

int ncnn_unit_plane_to_mat_u8(const ncnn::Mat& src, cv::Mat& dst)
{
    if (src.empty() || src.dims != 3 || src.c != 1)
    {
        return -1;
    }

    dst.create(src.h, src.w, CV_8UC1);
    printf("y_sr_image_addr: %p\n",dst.data);
    if (dst.empty())
    {
        return -1;
    }

    const float* srcptr = (const float*)src.channel(0);
    for (int y = 0; y < src.h; y++)
    {
    	printf("line: %d\n",y);
        uchar* row = dst.ptr(y);
        for (int x = 0; x < src.w; x++)
        {
            row[x] = float_to_u8(srcptr[y * src.w + x] * 255.f);
        }
    }

    return 0;
}

int upsample_chroma_on_npu_mat(const cv::Mat& chroma, ncnn::Mat& chroma_up, int scale)
{
    if (scale != 2 && scale != 4)
    {
        printf("[ERR][sesr] unsupported chroma upsample scale=%d\n", scale);
        return -1;
    }

    ncnn::Mat current_plane;
    if (mat_u8_to_ncnn_plane(chroma, current_plane) != 0)
    {
        printf("[ERR][sesr] prepare chroma plane failed\n");
        return -1;
    }

    const int stages = scale == 4 ? 2 : 1;
    for (int stage = 0; stage < stages; stage++)
    {
        ncnn::Mat shuffle_in;
        ncnn::Mat next_plane;

        if (replicate_plane_for_pixelshuffle(current_plane, shuffle_in) != 0)
        {
            printf("[ERR][sesr] stage%d chroma replicate failed\n", stage);
            return -1;
        }

        if (pixelshuffle_x2(shuffle_in, next_plane) != 0)
        {
            printf("[ERR][sesr] stage%d chroma pixelshuffle failed\n", stage);
            return -1;
        }

        current_plane = next_plane;
    }

    chroma_up = current_plane;
    return 0;
}

int ncnn_plane_float_to_bgr_channel_u8(const ncnn::Mat& src, cv::Mat& bgr, int bgr_channel)
{
    if (src.empty() || src.dims != 3 || src.c != 1 || src.elemsize != (size_t)4u || bgr.empty() || bgr.type() != CV_8UC3)
    {
        return -1;
    }

    if (src.w != bgr.cols || src.h != bgr.rows || bgr_channel < 0 || bgr_channel > 2)
    {
        return -1;
    }

    const float* srcptr = (const float*)src.channel(0);
    for (int y = 0; y < src.h; y++)
    {
        uchar* row = bgr.ptr(y);
        for (int x = 0; x < src.w; x++)
        {
            row[x * 3 + bgr_channel] = float_to_u8(srcptr[(size_t)y * src.w + x]);
        }
    }

    return 0;
}

int alu_scalar_add(const ncnn::Mat& src, float beta, ncnn::Mat& dst)
{
    if (src.empty() || src.dims != 3 || src.c != 1 || src.elemsize != (size_t)4u)
    {
        return -1;
    }

    dst.create(src.w, src.h, 1, (size_t)4u);
    if (dst.empty())
    {
        return -1;
    }

    ncnn::Mat& src_cast = const_cast<ncnn::Mat&>(src);
    accel::hw_math alu;
    alu.beta = beta;
    alu.add_src0_sel = ADD_SRC0_FROM_DMA;
    alu.add_src1_sel = ADD_SRC1_FROM_BETA;
    alu.add_en = 1;
    alu.math_forward(src_cast, dst);
    return 0;
}

int alu_scalar_mul(const ncnn::Mat& src, float alpha, ncnn::Mat& dst)
{
    if (src.empty() || src.dims != 3 || src.c != 1 || src.elemsize != (size_t)4u)
    {
        return -1;
    }

    dst.create(src.w, src.h, 1, (size_t)4u);
    if (dst.empty())
    {
        return -1;
    }

    ncnn::Mat& src_cast = const_cast<ncnn::Mat&>(src);
    accel::hw_math alu;
    alu.alpha = alpha;
    alu.mul_src1_sel = MUL_SRC1_FROM_ALPHA;
    alu.mul_en = 1;
    alu.math_forward(src_cast, dst);
    return 0;
}

int alu_add(const ncnn::Mat& a, const ncnn::Mat& b, ncnn::Mat& dst)
{
    if (a.empty() || b.empty() || a.dims != 3 || b.dims != 3 ||
        a.c != 1 || b.c != 1 || a.w != b.w || a.h != b.h ||
        a.elemsize != (size_t)4u || b.elemsize != (size_t)4u)
    {
        return -1;
    }

    dst.create(a.w, a.h, 1, (size_t)4u);
    if (dst.empty())
    {
        return -1;
    }

    ncnn::Mat& a_cast = const_cast<ncnn::Mat&>(a);
    ncnn::Mat& b_cast = const_cast<ncnn::Mat&>(b);
    accel::hw_math alu;
    alu.add_src0_sel = ADD_SRC0_FROM_DMA;
    alu.add_src1_sel = ADD_SRC1_FROM_DMA;
    alu.add_en = 1;
    alu.math_forward(a_cast, b_cast, dst);
    return 0;
}

int compose_bgr_on_npu_alu(const ncnn::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
#if defined(FORWARD_ON_NPU_ALU) && defined(FORWARD_ON_NPU_PIXELSHUFFLE)
    ncnn::Mat y255;
    ncnn::Mat cb_up;
    ncnn::Mat cr_up;
    ncnn::Mat cb_delta;
    ncnn::Mat cr_delta;
    ncnn::Mat b_term;
    ncnn::Mat g_cb_term;
    ncnn::Mat g_cr_term;
    ncnn::Mat g_y_cb;
    ncnn::Mat r_term;
    ncnn::Mat b_plane;
    ncnn::Mat g_plane;
    ncnn::Mat r_plane;

    if (y_sr.empty() || y_sr.dims != 3 || y_sr.c != 1 || y_sr.elemsize != (size_t)4u)
    {
        return -1;
    }

    if (upsample_chroma_on_npu_mat(cb_lr, cb_up, scale) != 0 || upsample_chroma_on_npu_mat(cr_lr, cr_up, scale) != 0)
    {
        return -1;
    }

    if (cb_up.w != y_sr.w || cb_up.h != y_sr.h || cr_up.w != y_sr.w || cr_up.h != y_sr.h)
    {
        return -1;
    }

    if (alu_scalar_mul(y_sr, 255.f, y255) != 0 ||
        alu_scalar_add(cb_up, -128.f, cb_delta) != 0 ||
        alu_scalar_add(cr_up, -128.f, cr_delta) != 0)
    {
        return -1;
    }

    if (alu_scalar_mul(cb_delta, 1.772f, b_term) != 0 ||
        alu_add(y255, b_term, b_plane) != 0)
    {
        return -1;
    }

    if (alu_scalar_mul(cr_delta, 1.402f, r_term) != 0 ||
        alu_add(y255, r_term, r_plane) != 0)
    {
        return -1;
    }

    if (alu_scalar_mul(cb_delta, -0.344136f, g_cb_term) != 0 ||
        alu_add(y255, g_cb_term, g_y_cb) != 0 ||
        alu_scalar_mul(cr_delta, -0.714136f, g_cr_term) != 0 ||
        alu_add(g_y_cb, g_cr_term, g_plane) != 0)
    {
        return -1;
    }

    bgr.create(y_sr.h, y_sr.w, CV_8UC3);
    if (bgr.empty())
    {
        return -1;
    }

    if (ncnn_plane_float_to_bgr_channel_u8(b_plane, bgr, 0) != 0 ||
        ncnn_plane_float_to_bgr_channel_u8(g_plane, bgr, 1) != 0 ||
        ncnn_plane_float_to_bgr_channel_u8(r_plane, bgr, 2) != 0)
    {
        return -1;
    }

    return 0;
#else
    (void)y_sr;
    (void)cb_lr;
    (void)cr_lr;
    (void)scale;
    (void)bgr;
    return -1;
#endif
}

} // namespace


void print_performance(u32 preprocess_finish_time,u32 inference_start_time)
{
    printf("totoal_cycles: %d\n"
           //"gemm_cycles: %d\n"
           //"alu_cycles: %d\n"
           //"pixelshuffle_cycles: %d\n"
           //"image_cycles: %d\n"
    	   //"preprocess_cycles: %d\n"
    		"npu_cycles: %d\n",
           //"image_axi_wait_cycles: %d\n",
           TOTAL_CYCLES - inference_start_time,
           //GEMM_CYCLES,
           //ALU_CYCLES,
           //PIXELSHUFFLE_CYCLES,
           //IMAGE_CYCLES,
		   //preprocess_finish_time-inference_start_time,
		   npu_cycles
           //IMAGE_AXI_WAIT_CYCLES
		   );
}

// SESR 推理函数
int sesr_inference()
{
#ifdef SESR_VERBOSE_LOG
    printf("[DBG][sesr] enter sesr_inference\n");
#endif

    // 1️⃣ 初始化网络
    ncnn::Net sesr;
    if (load_sesr_model(sesr) != 0)
    {
        return -1;
    }
#if defined(SESR_E2E_COMMAND) || defined(SESR_FIXED_COMMAND_TABLE)
    const ncnn::FixedParamTable* e2e_command_table = sesr_fixed_param_table(SESR_SCALE);
    if (!e2e_command_table)
    {
        printf("[ERR][sesr] e2e fixed command table not found for scale x%d\n", SESR_SCALE);
        return -1;
    }
#endif
    //perf.load_model_ms = //perf_ms_since(t);

    // 2️⃣ 读取图片
    //t = perf_now();
    cv::Mat img(INPUT_IMAGE_H, INPUT_IMAGE_W, INPUT_IMAGE_C,(void*) input_image_rgb);
    if (img.empty())
    {
         //printf("[ERR][sesr] failed to load image: %s\n", IMAGE_PATH);
        return -1;
    }

    int img_w = img.cols;
    int img_h = img.rows;
#ifdef SESR_VERBOSE_LOG
    printf("[DBG][sesr] image size: w=%d h=%d\n", img_w, img_h);
#endif

    u32 t_start_inference=TOTAL_CYCLES;
    npu_cycles=0;

    // 3️⃣ 预处理：BGR->YCbCr 和 Y 输入 tensor 打包。
    YCbCrImage input_ycbcr;
    ncnn::Mat y_input;
#ifdef FORWARD_ON_NPU_IMAGE
    if (bgr_to_ycbcr_and_y_input_npu(img, input_ycbcr, y_input) != 0)
    {
        printf("[ERR][sesr] NPU BGR -> YCbCr/Y-input failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("NPU", "preprocess-bgr-ycbcr-yinput", "input=%dx%d output_y=%dx%d save_y=%d",
                     img_w, img_h, y_input.w, y_input.h, SESR_PREPROCESS_SAVE_Y ? 1 : 0);
#else
    if (bgr_to_ycbcr_and_y_input(img, input_ycbcr, y_input) != 0)
    {
        printf("[ERR][sesr] fused BGR -> YCbCr/Y-input failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("CPU", "preprocess-bgr-ycbcr-yinput", "input=%dx%d output_y=%dx%d save_y=1", img_w, img_h, y_input.w, y_input.h);
#endif
    u32 preprocess_finish_time=TOTAL_CYCLES;
    cv::Mat y_inputu8;





#ifdef SESR_VERBOSE_LOG
    printf("[DBG][sesr] YCbCr input size: w=%d h=%d\n", input_ycbcr.width(), input_ycbcr.height());
#endif

    // 4️⃣ 只有 Y 通道走 SESR 推理，输出 tensor 直接交给后处理。
    ncnn::Mat y_sr;
    cv::Mat result;
#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU)
    accel::hw_sesr_tail_add_stream_prepare(input_ycbcr.width(), input_ycbcr.height(),
                                           y_input, input_ycbcr.cb, input_ycbcr.cr, SESR_SCALE, result);
#endif
#if defined(SESR_E2E_COMMAND) || defined(SESR_FIXED_COMMAND_TABLE)
    if (run_sesr_fixed_command(sesr, *e2e_command_table, y_input, y_sr) != 0)
#else
    if (run_sesr_on_y_input(sesr, y_input, y_sr) != 0)
#endif
    {
#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU)
        accel::hw_sesr_tail_add_stream_clear();
#endif
        return -1;
    }

#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ON_NPU)
    if (y_sr.w != input_ycbcr.width() || y_sr.h != input_ycbcr.height() || y_sr.c != SESR_SCALE * SESR_SCALE)
    {
        printf("[ERR][sesr] invalid fused-tail blob: input=(%d,%d) tail=(%d,%d,%d) expected_c=%d\n",
               input_ycbcr.width(), input_ycbcr.height(), y_sr.w, y_sr.h, y_sr.c, SESR_SCALE * SESR_SCALE);
        return -1;
    }
    const int scale_w = SESR_SCALE;
    const int scale_h = SESR_SCALE;
#else
    if (y_sr.w % input_ycbcr.width() != 0 || y_sr.h % input_ycbcr.height() != 0)
    {
        printf("[ERR][sesr] invalid Y scale: input=(%d,%d) output=(%d,%d)\n",
               input_ycbcr.width(), input_ycbcr.height(), y_sr.w, y_sr.h);
        return -1;
    }

    const int scale_w = y_sr.w / input_ycbcr.width();
    const int scale_h = y_sr.h / input_ycbcr.height();
#endif
    if (scale_w != scale_h || (scale_w != 2 && scale_w != 4))
    {
        printf("[ERR][sesr] unsupported Y scale: scale_w=%d scale_h=%d\n", scale_w, scale_h);
        return -1;
    }

    if (scale_w != SESR_SCALE)
    {
        printf("[ERR][sesr] model output scale x%d does not match SESR_SCALE x%d\n", scale_w, SESR_SCALE);
        return -1;
    }

#ifdef SESR_VERBOSE_LOG
    printf("[DBG][sesr] effective scale: x%d\n", scale_w);
#endif
    cv::Mat y_sr_image;
#ifdef SESR_SAVE_IMAGES
    #if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ADD_ON_NPU) || defined(SESR_FUSED_TAIL_ON_NPU)
        (void)y_sr_image;
    #else
        //t = perf_now();
        printf("start ncnn_unit_plane_to_mat_u8\n");
        if (ncnn_unit_plane_to_mat_u8(y_sr, y_sr_image) != 0)
        {
            printf("[ERR][sesr] export Y SR image failed\n");
            return -1;
        }
    #endif
#endif
    // 5️⃣ 后处理：完整 NPU 模式走专用图像硬件；旧 ALU/PixelShuffle 路径保留用于消融。
#if defined(SESR_FUSED_TAIL_STREAM_ADD_ON_NPU)
    accel::hw_sesr_tail_add_stream_wait();
    if (result.empty())
    {
        printf("[ERR][sesr] NPU stream tail-add produced empty result\n");
        return -1;
    }
    SESR_TRACE_STAGE("NPU", "tail-add-stream-conv-pixelshuffle-clip-ycbcr-bgr", "scale=x%d output=%dx%d", scale_w, result.cols, result.rows);
#elif defined(SESR_FUSED_TAIL_ADD_ON_NPU)
#ifdef SESR_FUSED_TAIL_REF_CHECK
    {
        cv::Mat tail_add_ref;
        if (compose_bgr_from_tail_add_blob_cpu_ref(y_sr, y_input, input_ycbcr.cb, input_ycbcr.cr, scale_w, tail_add_ref) != 0)
        {
            printf("[ERR][sesr] fused tail-add CPU reference failed\n");
            return -1;
        }
    }
#endif
    if (accel::hw_sesr_tail_add_fused_to_bgr(y_sr, y_input, input_ycbcr.cb, input_ycbcr.cr, scale_w, result) != 0)    {
        printf("[ERR][sesr] NPU fused tail-add postprocess failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("NPU", "tail-add-fused-add-pixelshuffle-clip-ycbcr-bgr", "scale=x%d output=%dx%d", scale_w, result.cols, result.rows);
#elif defined(SESR_FUSED_TAIL_ON_NPU)
#ifdef SESR_FUSED_TAIL_REF_CHECK
    {
        cv::Mat tail_ref;
        if (compose_bgr_from_tail_blob_cpu_ref(y_sr, input_ycbcr.cb, input_ycbcr.cr, scale_w, tail_ref) != 0)
        {
            printf("[ERR][sesr] fused tail CPU reference failed\n");
            return -1;
        }
    }
#endif
    if (accel::hw_sesr_tail_fused_to_bgr(y_sr, input_ycbcr.cb, input_ycbcr.cr, scale_w, result) != 0)
    {
        printf("[ERR][sesr] NPU fused tail postprocess failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("NPU", "tail-fused-pixelshuffle-clip-ycbcr-bgr", "scale=x%d output=%dx%d", scale_w, result.cols, result.rows);
#elif defined(FORWARD_ON_NPU_IMAGE)
    if (compose_bgr_from_y_float_and_lr_chroma_npu(y_sr, input_ycbcr.cb, input_ycbcr.cr, scale_w, result) != 0)
    {
        printf("[ERR][sesr] NPU dedicated postprocess failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("NPU", "postprocess-dedicated-ycbcr-bgr", "scale=x%d output=%dx%d", scale_w, result.cols, result.rows);
#elif defined(SESR_POSTPROCESS_ON_NPU_ALU)
    if (compose_bgr_on_npu_alu(y_sr, input_ycbcr.cb, input_ycbcr.cr, scale_w, result) != 0)
    {
        printf("[ERR][sesr] NPU ALU postprocess failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("NPU", "postprocess-pixelshuffle-alu-ycbcr-bgr", "scale=x%d output=%dx%d", scale_w, result.cols, result.rows);
#else
    if (compose_bgr_from_y_float_and_lr_chroma(y_sr, input_ycbcr.cb, input_ycbcr.cr, scale_w, result) != 0)
    {
        printf("[ERR][sesr] fused chroma upsample + YCbCr -> BGR failed\n");
        return -1;
    }
    SESR_TRACE_STAGE("CPU", "postprocess-fused-ycbcr-bgr", "scale=x%d output=%dx%d", scale_w, result.cols, result.rows);
#endif


    print_performance(preprocess_finish_time,t_start_inference);
    printf("SR_image_addr: %p\n\n",result.data);


#ifdef SESR_VERBOSE_LOG
    printf("[DBG][sesr] inference done\n");
#endif
    return 0;
}

} // namespace demo
