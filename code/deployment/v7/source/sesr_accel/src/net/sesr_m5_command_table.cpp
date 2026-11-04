
#include "sesr_m5_command_table.h"

namespace demo {

namespace x2_fused {
static const ncnn::FixedBlob blobs[] = {
    {"data", 0, 1},
    {"data_to_sub", 1, 2},
    {"data_skip", 1, 12},
    {"data_centered", 2, 3},
    {"feat0", 3, 4},
    {"feat_run", 4, 5},
    {"feat_short", 4, 10},
    {"mid1_act_out", 5, 6},
    {"mid2_act_out", 6, 7},
    {"mid3_act_out", 7, 8},
    {"mid4_act_out", 8, 9},
    {"mid5_act_out", 9, 10},
    {"res1", 10, 11},
    {"output_conv_out", 11, 12},
    {"res2_out", 12, 13},
    {"ps0_out", 13, 14},
    {"output", 14, -1},
    {"__unused_blob_17", -1, -1},
};

static const int layer_0_bottoms[] = {0};
static const int layer_0_tops[] = {0};
static const ncnn::FixedParamValue layer_0_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_1_bottoms[] = {0};
static const int layer_1_tops[] = {1, 2};
static const ncnn::FixedParamValue layer_1_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_2_bottoms[] = {1};
static const int layer_2_tops[] = {3};
static const ncnn::FixedParamValue layer_2_params[] = {
    {0, 0, 0, 0.0f},
    {1, 0, 1, 0.0f},
    {2, 1, 0, -0.5f},
};
static const int layer_3_bottoms[] = {3};
static const int layer_3_tops[] = {4};
static const ncnn::FixedParamValue layer_3_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 5, 0.0f},
    {11, 0, 5, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 2, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 800, 0.0f},
    {8, 0, 2, 0.0f},
};
static const int layer_4_bottoms[] = {4};
static const int layer_4_tops[] = {5, 6};
static const ncnn::FixedParamValue layer_4_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_5_bottoms[] = {5};
static const int layer_5_tops[] = {7};
static const ncnn::FixedParamValue layer_5_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_6_bottoms[] = {7};
static const int layer_6_tops[] = {8};
static const ncnn::FixedParamValue layer_6_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_7_bottoms[] = {8};
static const int layer_7_tops[] = {9};
static const ncnn::FixedParamValue layer_7_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_8_bottoms[] = {9};
static const int layer_8_tops[] = {10};
static const ncnn::FixedParamValue layer_8_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_9_bottoms[] = {10};
static const int layer_9_tops[] = {11};
static const ncnn::FixedParamValue layer_9_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_10_bottoms[] = {11, 6};
static const int layer_10_tops[] = {12};
static const ncnn::FixedParamValue layer_10_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_11_bottoms[] = {12};
static const int layer_11_tops[] = {13};
static const ncnn::FixedParamValue layer_11_params[] = {
    {0, 0, 4, 0.0f},
    {1, 0, 5, 0.0f},
    {11, 0, 5, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 2, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 3200, 0.0f},
    {8, 0, 2, 0.0f},
};
static const int layer_12_bottoms[] = {13, 2};
static const int layer_12_tops[] = {14};
static const ncnn::FixedParamValue layer_12_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_13_bottoms[] = {14};
static const int layer_13_tops[] = {15};
static const ncnn::FixedParamValue layer_13_params[] = {
    {0, 0, 2, 0.0f},
    {1, 0, 0, 0.0f},
};
static const int layer_14_bottoms[] = {15};
static const int layer_14_tops[] = {16};
static const ncnn::FixedParamValue layer_14_params[] = {
    {0, 1, 0, 0.0f},
    {1, 1, 0, 1.0f},
};

static const ncnn::FixedLayer layers[] = {
    {0, "Input", "input", 0, 1, layer_0_bottoms, layer_0_tops, 0, layer_0_params},
    {1, "Split", "split_input", 1, 2, layer_1_bottoms, layer_1_tops, 0, layer_1_params},
    {2, "BinaryOp", "sub_05", 1, 1, layer_2_bottoms, layer_2_tops, 3, layer_2_params},
    {3, "Convolution", "input_block", 1, 1, layer_3_bottoms, layer_3_tops, 11, layer_3_params},
    {1, "Split", "split_feat0", 1, 2, layer_4_bottoms, layer_4_tops, 0, layer_4_params},
    {3, "Convolution", "mid1_conv", 1, 1, layer_5_bottoms, layer_5_tops, 12, layer_5_params},
    {3, "Convolution", "mid2_conv", 1, 1, layer_6_bottoms, layer_6_tops, 12, layer_6_params},
    {3, "Convolution", "mid3_conv", 1, 1, layer_7_bottoms, layer_7_tops, 12, layer_7_params},
    {3, "Convolution", "mid4_conv", 1, 1, layer_8_bottoms, layer_8_tops, 12, layer_8_params},
    {3, "Convolution", "mid5_conv", 1, 1, layer_9_bottoms, layer_9_tops, 12, layer_9_params},
    {2, "BinaryOp", "add_long_res", 2, 1, layer_10_bottoms, layer_10_tops, 1, layer_10_params},
    {3, "Convolution", "output_block", 1, 1, layer_11_bottoms, layer_11_tops, 11, layer_11_params},
    {2, "BinaryOp", "add_global", 2, 1, layer_12_bottoms, layer_12_tops, 1, layer_12_params},
    {6, "PixelShuffle", "pixelshuffle_0", 1, 1, layer_13_bottoms, layer_13_tops, 2, layer_13_params},
    {7, "Clip", "clip_out", 1, 1, layer_14_bottoms, layer_14_tops, 2, layer_14_params},
};

static const ncnn::FixedParamTable table = {15, 18, layers, blobs, 0, 16, 13};
static const int scale = 2;
} // namespace x2_fused

namespace x4_fused {
static const ncnn::FixedBlob blobs[] = {
    {"data", 0, 1},
    {"data_to_sub", 1, 2},
    {"data_skip", 1, 12},
    {"data_centered", 2, 3},
    {"feat0", 3, 4},
    {"feat_run", 4, 5},
    {"feat_short", 4, 10},
    {"mid1_act_out", 5, 6},
    {"mid2_act_out", 6, 7},
    {"mid3_act_out", 7, 8},
    {"mid4_act_out", 8, 9},
    {"mid5_act_out", 9, 10},
    {"res1", 10, 11},
    {"output_conv_out", 11, 12},
    {"res2_out", 12, 13},
    {"ps0_out", 13, 14},
    {"ps1_out", 14, 15},
    {"output", 15, -1},
    {"__unused_blob_18", -1, -1},
};

static const int layer_0_bottoms[] = {0};
static const int layer_0_tops[] = {0};
static const ncnn::FixedParamValue layer_0_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_1_bottoms[] = {0};
static const int layer_1_tops[] = {1, 2};
static const ncnn::FixedParamValue layer_1_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_2_bottoms[] = {1};
static const int layer_2_tops[] = {3};
static const ncnn::FixedParamValue layer_2_params[] = {
    {0, 0, 0, 0.0f},
    {1, 0, 1, 0.0f},
    {2, 1, 0, -0.5f},
};
static const int layer_3_bottoms[] = {3};
static const int layer_3_tops[] = {4};
static const ncnn::FixedParamValue layer_3_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 5, 0.0f},
    {11, 0, 5, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 2, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 800, 0.0f},
    {8, 0, 2, 0.0f},
};
static const int layer_4_bottoms[] = {4};
static const int layer_4_tops[] = {5, 6};
static const ncnn::FixedParamValue layer_4_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_5_bottoms[] = {5};
static const int layer_5_tops[] = {7};
static const ncnn::FixedParamValue layer_5_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_6_bottoms[] = {7};
static const int layer_6_tops[] = {8};
static const ncnn::FixedParamValue layer_6_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_7_bottoms[] = {8};
static const int layer_7_tops[] = {9};
static const ncnn::FixedParamValue layer_7_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_8_bottoms[] = {9};
static const int layer_8_tops[] = {10};
static const ncnn::FixedParamValue layer_8_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_9_bottoms[] = {10};
static const int layer_9_tops[] = {11};
static const ncnn::FixedParamValue layer_9_params[] = {
    {0, 0, 32, 0.0f},
    {1, 0, 3, 0.0f},
    {11, 0, 3, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 1, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 9216, 0.0f},
    {8, 0, 2, 0.0f},
    {9, 0, 1, 0.0f},
};
static const int layer_10_bottoms[] = {11, 6};
static const int layer_10_tops[] = {12};
static const ncnn::FixedParamValue layer_10_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_11_bottoms[] = {12};
static const int layer_11_tops[] = {13};
static const ncnn::FixedParamValue layer_11_params[] = {
    {0, 0, 16, 0.0f},
    {1, 0, 5, 0.0f},
    {11, 0, 5, 0.0f},
    {2, 0, 1, 0.0f},
    {12, 0, 1, 0.0f},
    {3, 0, 1, 0.0f},
    {13, 0, 1, 0.0f},
    {4, 0, 2, 0.0f},
    {5, 0, 1, 0.0f},
    {6, 0, 12800, 0.0f},
    {8, 0, 2, 0.0f},
};
static const int layer_12_bottoms[] = {13, 2};
static const int layer_12_tops[] = {14};
static const ncnn::FixedParamValue layer_12_params[] = {
    {0, 0, 0, 0.0f},
};
static const int layer_13_bottoms[] = {14};
static const int layer_13_tops[] = {15};
static const ncnn::FixedParamValue layer_13_params[] = {
    {0, 0, 2, 0.0f},
    {1, 0, 1, 0.0f},
};
static const int layer_14_bottoms[] = {15};
static const int layer_14_tops[] = {16};
static const ncnn::FixedParamValue layer_14_params[] = {
    {0, 0, 2, 0.0f},
    {1, 0, 0, 0.0f},
};
static const int layer_15_bottoms[] = {16};
static const int layer_15_tops[] = {17};
static const ncnn::FixedParamValue layer_15_params[] = {
    {0, 1, 0, 0.0f},
    {1, 1, 0, 1.0f},
};

static const ncnn::FixedLayer layers[] = {
    {0, "Input", "input", 0, 1, layer_0_bottoms, layer_0_tops, 0, layer_0_params},
    {1, "Split", "split_input", 1, 2, layer_1_bottoms, layer_1_tops, 0, layer_1_params},
    {2, "BinaryOp", "sub_05", 1, 1, layer_2_bottoms, layer_2_tops, 3, layer_2_params},
    {3, "Convolution", "input_block", 1, 1, layer_3_bottoms, layer_3_tops, 11, layer_3_params},
    {1, "Split", "split_feat0", 1, 2, layer_4_bottoms, layer_4_tops, 0, layer_4_params},
    {3, "Convolution", "mid1_conv", 1, 1, layer_5_bottoms, layer_5_tops, 12, layer_5_params},
    {3, "Convolution", "mid2_conv", 1, 1, layer_6_bottoms, layer_6_tops, 12, layer_6_params},
    {3, "Convolution", "mid3_conv", 1, 1, layer_7_bottoms, layer_7_tops, 12, layer_7_params},
    {3, "Convolution", "mid4_conv", 1, 1, layer_8_bottoms, layer_8_tops, 12, layer_8_params},
    {3, "Convolution", "mid5_conv", 1, 1, layer_9_bottoms, layer_9_tops, 12, layer_9_params},
    {2, "BinaryOp", "add_long_res", 2, 1, layer_10_bottoms, layer_10_tops, 1, layer_10_params},
    {3, "Convolution", "output_block", 1, 1, layer_11_bottoms, layer_11_tops, 11, layer_11_params},
    {2, "BinaryOp", "add_global", 2, 1, layer_12_bottoms, layer_12_tops, 1, layer_12_params},
    {6, "PixelShuffle", "pixelshuffle_0", 1, 1, layer_13_bottoms, layer_13_tops, 2, layer_13_params},
    {6, "PixelShuffle", "pixelshuffle_1", 1, 1, layer_14_bottoms, layer_14_tops, 2, layer_14_params},
    {7, "Clip", "clip_out", 1, 1, layer_15_bottoms, layer_15_tops, 2, layer_15_params},
};

static const ncnn::FixedParamTable table = {16, 19, layers, blobs, 0, 17, 13};
static const int scale = 4;
} // namespace x4_fused

const ncnn::FixedParamTable* sesr_fixed_param_table(int scale)
{
    if (scale == 2) return &x2_fused::table;
    if (scale == 4) return &x4_fused::table;
    return 0;
}

const char* sesr_fixed_param_table_name(int scale)
{
    if (scale == 2) return "x2_fused";
    if (scale == 4) return "x4_fused";
    return "unknown";
}

} // namespace demo
