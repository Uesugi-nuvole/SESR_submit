// Minimal layer registry for the SESR simulation target.

static const layer_registry_entry layer_registry[] = {
#if NCNN_STRING
{"Input", Input_final_layer_creator},
{"Split", Split_final_layer_creator},
{"BinaryOp", BinaryOp_final_layer_creator},
{"Convolution", Convolution_final_layer_creator},
{"Padding", Padding_final_layer_creator},
{"ReLU", ReLU_final_layer_creator},
{"PixelShuffle", PixelShuffle_final_layer_creator},
{"Clip", Clip_final_layer_creator},
#else
{Input_final_layer_creator},
{Split_final_layer_creator},
{BinaryOp_final_layer_creator},
{Convolution_final_layer_creator},
{Padding_final_layer_creator},
{ReLU_final_layer_creator},
{PixelShuffle_final_layer_creator},
{Clip_final_layer_creator},
#endif
};
