#ifndef __CONFIG_H_
#define __CONFIG_H_

// ---------------------------------------------------------------------------
// E2E / fixed command table
// ---------------------------------------------------------------------------
//#define SESR_E2E_COMMAND
//#define SESR_FIXED_COMMAND_TABLE

// ---------------------------------------------------------------------------
// SESR inference mode
// ---------------------------------------------------------------------------
#define SESR_FUSED_TAIL_STREAM_ADD_ON_NPU   // Streamed tail-add NPU postprocess (default)
#define SESR_FORWARD_ON_NPU                 // Master switch: run entire network on NPU
//#define SESR_FORWARD_ON_CPU               // Override: run on CPU
//#define SESR_POSTPROCESS_ON_NPU_ALU       // Use ALU for YCbCr→BGR postprocess

// ---------------------------------------------------------------------------
// Model scale factor
// ---------------------------------------------------------------------------
#ifndef SESR_SCALE
#define SESR_SCALE 4                     // 2 → 640×360 → 1280×720; 4 → 320×180 → 1280×720


#endif

#if defined(SESR_FORWARD_ON_NPU)
#define FORWARD_ON_NPU
#elif defined(SESR_FORWARD_ON_CPU)
#define FORWARD_ON_CPU
#endif

#if !defined(FORWARD_ON_CPU) && !defined(FORWARD_ON_NPU) && \
    !defined(FORWARD_ON_NPU_CONV) && !defined(FORWARD_ON_NPU_ALU) && \
    !defined(FORWARD_ON_NPU_PIXELSHUFFLE) && !defined(FORWARD_ON_NPU_IMAGE)
#define FORWARD_ON_CPU
#endif

#ifdef NCNN_TOOLS
#undef FORWARD_ON_NPU
#undef FORWARD_ON_NPU_CONV
#undef FORWARD_ON_NPU_ALU
#undef FORWARD_ON_NPU_PIXELSHUFFLE
#undef FORWARD_ON_NPU_IMAGE
#define FORWARD_ON_CPU
#endif

#ifdef FORWARD_ON_NPU
#define FORWARD_ON_NPU_CONV
#define FORWARD_ON_NPU_ALU
#define FORWARD_ON_NPU_PIXELSHUFFLE
#define FORWARD_ON_NPU_IMAGE
#endif

#ifdef FORWARD_ON_CPU
#define FORWARD_ON_CPU_CONV
#define FORWARD_ON_CPU_ALU
#endif

#if !defined(FORWARD_ON_NPU_CONV) && !defined(FORWARD_ON_CPU_CONV)
#define FORWARD_ON_CPU_CONV
#endif
#if !defined(FORWARD_ON_NPU_ALU) && !defined(FORWARD_ON_CPU_ALU)
#define FORWARD_ON_CPU_ALU
#endif

// ---------------------------------------------------------------------------
// ncnn / simulation settings
// ---------------------------------------------------------------------------
#define NCNN_MAT_ALIGN_BYTES 32
#define CHECK_ALIGN


// ---------------------------------------------------------------------------
// Debug & trace
// ---------------------------------------------------------------------------
//#define SESR_VERBOSE_LOG          // Enable per-layer shape + layer name logging

#if defined(SESR_VERBOSE_LOG)
#define PRINT_SHAPE
#define PRINT_LAYER
#endif

// ---------------------------------------------------------------------------
// Bus & DMA config
// ---------------------------------------------------------------------------
#define AXI_LITE_DATA_SUPPORT
#define DMA_SUPPORT
#define ACCMEM_GAP 1000

// ---------------------------------------------------------------------------
// NPU buffer dimensions
// ---------------------------------------------------------------------------
#define IFM_BUFFER_WIDTH 256
#define IFM_BUFFER_DEPTH (65536 / 4)
#define WGT_BUFFER_DEPTH (9 * 512)
#define OSCALE_BUFFER_DEPTH 1024

// ---------------------------------------------------------------------------
// AXI / DMA / clock parameters
// ---------------------------------------------------------------------------
#define ACCMEM_OUT_WIDTH 29
#define DMA_DATA_WIDTH 128
#define AXI_DATA_WIDTH 128
#define AXI_TRANSFER_BYTE (AXI_DATA_WIDTH / 8)
#define DMA_DATA_NUM (DMA_DATA_WIDTH / 32)
#define HW_FREQ_MHZ 500

// ---------------------------------------------------------------------------
// Accelerator module enable
// ---------------------------------------------------------------------------
#define INFERENCE_NET
#define USE_HW_ALU
#define USE_HW_PIXELSHUFFLE
#define USE_HW_GEMM
#define USE_HW_SESR_IMAGE

#endif
