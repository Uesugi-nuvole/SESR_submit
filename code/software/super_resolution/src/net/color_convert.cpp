#include "color_convert.h"

#include "../hw/hw_sesr_image.h"
#include <algorithm>
#include <stdint.h>
#include <string.h>

namespace demo {

static inline float clamp_255(float v)
{
    return std::min(std::max(v, 0.f), 255.f);
}

static inline unsigned char float_to_u8(float v)
{
    return (unsigned char)(clamp_255(v) + 0.5f);
}

static inline unsigned char float_unit_to_u8(float v)
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

static inline int clamp_int(int v, int lo, int hi)
{
    return std::min(std::max(v, lo), hi);
}

static inline unsigned char fixed_to_u8(int v)
{
    v = (v + (1 << 15)) >> 16;
    return (unsigned char)clamp_int(v, 0, 255);
}

static inline void ycbcr_fixed_to_bgr_store(int y_u8, int cb_u8, int cr_u8, uchar* bgr)
{
    const int yy = y_u8 << 16;
    const int cb = cb_u8 - 128;
    const int cr = cr_u8 - 128;

    const int r = yy + 91881 * cr;
    const int g = yy - 22554 * cb - 46802 * cr;
    const int b = yy + 116130 * cb;

    bgr[0] = fixed_to_u8(b);
    bgr[1] = fixed_to_u8(g);
    bgr[2] = fixed_to_u8(r);
}

struct ChromaFixedTerms
{
    int b;
    int g;
    int r;
};

struct YCbCrFixedTables
{
    int y_r[256];
    int y_g[256];
    int y_b[256];
    int cb_r[256];
    int cb_g[256];
    int cb_b[256];
    int cr_r[256];
    int cr_g[256];
    int cr_b[256];
};

static inline const YCbCrFixedTables& ycbcr_fixed_tables()
{
    static YCbCrFixedTables tables;
    static bool initialized = false;
    if (!initialized)
    {
        for (int i = 0; i < 256; i++)
        {
            tables.y_r[i] = 19595 * i;
            tables.y_g[i] = 38470 * i;
            tables.y_b[i] = 7471 * i;
            tables.cb_r[i] = -11058 * i;
            tables.cb_g[i] = -21710 * i;
            tables.cb_b[i] = 32768 * i;
            tables.cr_r[i] = 32768 * i;
            tables.cr_g[i] = -27439 * i;
            tables.cr_b[i] = -5329 * i;
        }
        initialized = true;
    }
    return tables;
}

static inline const ChromaFixedTerms& ycbcr_chroma_fixed_terms_table(int cb_u8, int cr_u8)
{
    static ChromaFixedTerms terms[256][256];
    static bool initialized = false;
    if (!initialized)
    {
        for (int cb_u = 0; cb_u < 256; cb_u++)
        {
            for (int cr_u = 0; cr_u < 256; cr_u++)
            {
                const int cb = cb_u - 128;
                const int cr = cr_u - 128;
                terms[cb_u][cr_u].b = 116130 * cb;
                terms[cb_u][cr_u].g = -22554 * cb - 46802 * cr;
                terms[cb_u][cr_u].r = 91881 * cr;
            }
        }
        initialized = true;
    }
    return terms[cb_u8][cr_u8];
}

static inline ChromaFixedTerms ycbcr_chroma_fixed_terms(int cb_u8, int cr_u8)
{
    const int cb = cb_u8 - 128;
    const int cr = cr_u8 - 128;

    ChromaFixedTerms terms;
    terms.b = 116130 * cb;
    terms.g = -22554 * cb - 46802 * cr;
    terms.r = 91881 * cr;
    return terms;
}

static inline void ycbcr_terms_to_bgr_store(int y_u8, const ChromaFixedTerms& terms, uchar* bgr)
{
    const int yy = y_u8 << 16;
    bgr[0] = fixed_to_u8(yy + terms.b);
    bgr[1] = fixed_to_u8(yy + terms.g);
    bgr[2] = fixed_to_u8(yy + terms.r);
}

static inline void bgr_to_ycbcr_fixed(unsigned char b, unsigned char g, unsigned char r, unsigned char& y, unsigned char& cb, unsigned char& cr)
{
    const YCbCrFixedTables& tables = ycbcr_fixed_tables();
    const int yy = tables.y_r[r] + tables.y_g[g] + tables.y_b[b];
    const int cbv = (128 << 16) + tables.cb_r[r] + tables.cb_g[g] + tables.cb_b[b];
    const int crv = (128 << 16) + tables.cr_r[r] + tables.cr_g[g] + tables.cr_b[b];
    y = fixed_to_u8(yy);
    cb = fixed_to_u8(cbv);
    cr = fixed_to_u8(crv);
}

void rgb_to_ycbcr_pixel(float r, float g, float b, float& y, float& cb, float& cr)
{
    y = 0.299f * r + 0.587f * g + 0.114f * b;
    cb = 128.f - 0.168736f * r - 0.331264f * g + 0.500000f * b;
    cr = 128.f + 0.500000f * r - 0.418688f * g - 0.081312f * b;
}

void ycbcr_to_rgb_pixel(float y, float cb, float cr, float& r, float& g, float& b)
{
    r = y + 1.402000f * (cr - 128.f);
    g = y - 0.344136f * (cb - 128.f) - 0.714136f * (cr - 128.f);
    b = y + 1.772000f * (cb - 128.f);
}

int bgr_to_ycbcr(const cv::Mat& bgr, YCbCrImage& ycbcr)
{
    if (bgr.empty() || bgr.type() != CV_8UC3)
    {
        return -1;
    }

    ycbcr.image_w = bgr.cols;
    ycbcr.image_h = bgr.rows;
    ycbcr.y.create(bgr.rows, bgr.cols, CV_8UC1);
    ycbcr.cb.create(bgr.rows, bgr.cols, CV_8UC1);
    ycbcr.cr.create(bgr.rows, bgr.cols, CV_8UC1);

    if (ycbcr.y.empty() || ycbcr.cb.empty() || ycbcr.cr.empty())
    {
        return -1;
    }

    for (int y = 0; y < bgr.rows; y++)
    {
        const uchar* bgr_row = bgr.ptr(y);
        uchar* y_row = ycbcr.y.ptr(y);
        uchar* cb_row = ycbcr.cb.ptr(y);
        uchar* cr_row = ycbcr.cr.ptr(y);

        for (int x = 0; x < bgr.cols; x++)
        {
            bgr_to_ycbcr_fixed(
                bgr_row[x * 3 + 0],
                bgr_row[x * 3 + 1],
                bgr_row[x * 3 + 2],
                y_row[x],
                cb_row[x],
                cr_row[x]);
        }
    }

    return 0;
}

int bgr_to_ycbcr_and_y_input(const cv::Mat& bgr, YCbCrImage& ycbcr, ncnn::Mat& y_input)
{
    if (bgr.empty() || bgr.type() != CV_8UC3)
    {
        return -1;
    }

    ycbcr.image_w = bgr.cols;
    ycbcr.image_h = bgr.rows;
    ycbcr.y.create(bgr.rows, bgr.cols, CV_8UC1);
    ycbcr.cb.create(bgr.rows, bgr.cols, CV_8UC1);
    ycbcr.cr.create(bgr.rows, bgr.cols, CV_8UC1);
    y_input.create(bgr.cols, bgr.rows, 1, (size_t)4u);

    if (ycbcr.y.empty() || ycbcr.cb.empty() || ycbcr.cr.empty() || y_input.empty())
    {
        return -1;
    }

    float* y_input_ptr = (float*)y_input.channel(0);
    for (int y = 0; y < bgr.rows; y++)
    {
        const uchar* bgr_row = bgr.ptr(y);
        uchar* y_row = ycbcr.y.ptr(y);
        uchar* cb_row = ycbcr.cb.ptr(y);
        uchar* cr_row = ycbcr.cr.ptr(y);
        float* y_input_row = y_input_ptr + (size_t)y * bgr.cols;

        for (int x = 0; x < bgr.cols; x++)
        {
            bgr_to_ycbcr_fixed(
                bgr_row[x * 3 + 0],
                bgr_row[x * 3 + 1],
                bgr_row[x * 3 + 2],
                y_row[x],
                cb_row[x],
                cr_row[x]);
            y_input_row[x] = (float)y_row[x] / 255.f;
        }
    }

    return 0;
}

int bgr_to_ycbcr_and_y_input_npu(const cv::Mat& bgr, YCbCrImage& ycbcr, ncnn::Mat& y_input, bool save_y)
{
    ycbcr.image_w = bgr.cols;
    ycbcr.image_h = bgr.rows;
    return accel::hw_sesr_preprocess_bgr_to_ycbcr_yinput(bgr, ycbcr.y, ycbcr.cb, ycbcr.cr, y_input, save_y);
}

int ycbcr_to_bgr(const YCbCrImage& ycbcr, cv::Mat& bgr)
{
    if (ycbcr.y.empty() || ycbcr.cb.empty() || ycbcr.cr.empty())
    {
        return -1;
    }

    if (ycbcr.y.type() != CV_8UC1 || ycbcr.cb.type() != CV_8UC1 || ycbcr.cr.type() != CV_8UC1)
    {
        return -1;
    }

    if (ycbcr.y.rows != ycbcr.cb.rows || ycbcr.y.rows != ycbcr.cr.rows ||
        ycbcr.y.cols != ycbcr.cb.cols || ycbcr.y.cols != ycbcr.cr.cols)
    {
        return -1;
    }

    bgr.create(ycbcr.y.rows, ycbcr.y.cols, CV_8UC3);
    if (bgr.empty())
    {
        return -1;
    }

    for (int y = 0; y < ycbcr.y.rows; y++)
    {
        const uchar* y_row = ycbcr.y.ptr(y);
        const uchar* cb_row = ycbcr.cb.ptr(y);
        const uchar* cr_row = ycbcr.cr.ptr(y);
        uchar* bgr_row = bgr.ptr(y);

        for (int x = 0; x < ycbcr.y.cols; x++)
        {
            float r;
            float g;
            float b;
            ycbcr_to_rgb_pixel((float)y_row[x], (float)cb_row[x], (float)cr_row[x], r, g, b);

            bgr_row[x * 3 + 0] = float_to_u8(b);
            bgr_row[x * 3 + 1] = float_to_u8(g);
            bgr_row[x * 3 + 2] = float_to_u8(r);
        }
    }

    return 0;
}

int compose_bgr_from_y_and_lr_chroma(const cv::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    if (scale != 2 && scale != 4)
    {
        return -1;
    }

    if (y_sr.empty() || cb_lr.empty() || cr_lr.empty())
    {
        return -1;
    }

    if (y_sr.type() != CV_8UC1 || cb_lr.type() != CV_8UC1 || cr_lr.type() != CV_8UC1)
    {
        return -1;
    }

    if (cb_lr.rows != cr_lr.rows || cb_lr.cols != cr_lr.cols)
    {
        return -1;
    }

    if (y_sr.rows != cb_lr.rows * scale || y_sr.cols != cb_lr.cols * scale)
    {
        return -1;
    }

    bgr.create(y_sr.rows, y_sr.cols, CV_8UC3);
    if (bgr.empty())
    {
        return -1;
    }

    for (int y = 0; y < y_sr.rows; y++)
    {
        const uchar* y_row = y_sr.ptr(y);
        const uchar* cb_row = cb_lr.ptr(y / scale);
        const uchar* cr_row = cr_lr.ptr(y / scale);
        uchar* bgr_row = bgr.ptr(y);

        for (int x = 0; x < y_sr.cols; x++)
        {
            const int yy = (int)y_row[x] << 16;
            const int cb = (int)cb_row[x / scale] - 128;
            const int cr = (int)cr_row[x / scale] - 128;

            const int r = yy + 91881 * cr;
            const int g = yy - 22554 * cb - 46802 * cr;
            const int b = yy + 116130 * cb;

            bgr_row[x * 3 + 0] = fixed_to_u8(b);
            bgr_row[x * 3 + 1] = fixed_to_u8(g);
            bgr_row[x * 3 + 2] = fixed_to_u8(r);
        }
    }

    return 0;
}

int compose_bgr_from_y_float_and_lr_chroma(const ncnn::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    if (scale != 2 && scale != 4)
    {
        return -1;
    }

    if (y_sr.empty() || cb_lr.empty() || cr_lr.empty())
    {
        return -1;
    }

    if (y_sr.dims != 3 || y_sr.c != 1 || y_sr.elemsize != (size_t)4u)
    {
        return -1;
    }

    if (cb_lr.type() != CV_8UC1 || cr_lr.type() != CV_8UC1)
    {
        return -1;
    }

    if (cb_lr.rows != cr_lr.rows || cb_lr.cols != cr_lr.cols)
    {
        return -1;
    }

    if (y_sr.h != cb_lr.rows * scale || y_sr.w != cb_lr.cols * scale)
    {
        return -1;
    }

    bgr.create(y_sr.h, y_sr.w, CV_8UC3);
    if (bgr.empty())
    {
        return -1;
    }

    const float* y_data = (const float*)y_sr.channel(0);

    if (scale == 4)
    {
        for (int ly = 0; ly < cb_lr.rows; ly++)
        {
            const uchar* cb_row = cb_lr.ptr(ly);
            const uchar* cr_row = cr_lr.ptr(ly);
            const int hy = ly * 4;
            const float* y_row0 = y_data + (size_t)(hy + 0) * y_sr.w;
            const float* y_row1 = y_data + (size_t)(hy + 1) * y_sr.w;
            const float* y_row2 = y_data + (size_t)(hy + 2) * y_sr.w;
            const float* y_row3 = y_data + (size_t)(hy + 3) * y_sr.w;
            uchar* bgr_row0 = bgr.ptr(hy + 0);
            uchar* bgr_row1 = bgr.ptr(hy + 1);
            uchar* bgr_row2 = bgr.ptr(hy + 2);
            uchar* bgr_row3 = bgr.ptr(hy + 3);

            for (int lx = 0; lx < cb_lr.cols; lx++)
            {
                const ChromaFixedTerms& terms = ycbcr_chroma_fixed_terms_table(cb_row[lx], cr_row[lx]);
                const int hx = lx * 4;

                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row0[hx + 0]), terms, bgr_row0 + (hx + 0) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row0[hx + 1]), terms, bgr_row0 + (hx + 1) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row0[hx + 2]), terms, bgr_row0 + (hx + 2) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row0[hx + 3]), terms, bgr_row0 + (hx + 3) * 3);

                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row1[hx + 0]), terms, bgr_row1 + (hx + 0) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row1[hx + 1]), terms, bgr_row1 + (hx + 1) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row1[hx + 2]), terms, bgr_row1 + (hx + 2) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row1[hx + 3]), terms, bgr_row1 + (hx + 3) * 3);

                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row2[hx + 0]), terms, bgr_row2 + (hx + 0) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row2[hx + 1]), terms, bgr_row2 + (hx + 1) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row2[hx + 2]), terms, bgr_row2 + (hx + 2) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row2[hx + 3]), terms, bgr_row2 + (hx + 3) * 3);

                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row3[hx + 0]), terms, bgr_row3 + (hx + 0) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row3[hx + 1]), terms, bgr_row3 + (hx + 1) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row3[hx + 2]), terms, bgr_row3 + (hx + 2) * 3);
                ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row3[hx + 3]), terms, bgr_row3 + (hx + 3) * 3);
            }
        }

        return 0;
    }

    for (int ly = 0; ly < cb_lr.rows; ly++)
    {
        const uchar* cb_row = cb_lr.ptr(ly);
        const uchar* cr_row = cr_lr.ptr(ly);
        const int hy = ly * 2;
        const float* y_row0 = y_data + (size_t)(hy + 0) * y_sr.w;
        const float* y_row1 = y_data + (size_t)(hy + 1) * y_sr.w;
        uchar* bgr_row0 = bgr.ptr(hy + 0);
        uchar* bgr_row1 = bgr.ptr(hy + 1);

        for (int lx = 0; lx < cb_lr.cols; lx++)
        {
            const ChromaFixedTerms& terms = ycbcr_chroma_fixed_terms_table(cb_row[lx], cr_row[lx]);
            const int hx = lx * 2;

            ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row0[hx + 0]), terms, bgr_row0 + (hx + 0) * 3);
            ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row0[hx + 1]), terms, bgr_row0 + (hx + 1) * 3);
            ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row1[hx + 0]), terms, bgr_row1 + (hx + 0) * 3);
            ycbcr_terms_to_bgr_store(float_unit_to_u8(y_row1[hx + 1]), terms, bgr_row1 + (hx + 1) * 3);
        }
    }

    return 0;
}

int compose_bgr_from_y_float_and_lr_chroma_npu(const ncnn::Mat& y_sr, const cv::Mat& cb_lr, const cv::Mat& cr_lr, int scale, cv::Mat& bgr)
{
    return accel::hw_sesr_postprocess_ycbcr_to_bgr(y_sr, cb_lr, cr_lr, scale, bgr);
}

} // namespace demo
