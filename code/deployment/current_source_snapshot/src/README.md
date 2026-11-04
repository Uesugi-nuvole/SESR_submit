# SESR 加速器

基于 Xilinx FPGA 的 SESR（超分辨率）推理引擎，运行在 MicroBlaze 上，通过自定义 NPU 硬件加速器实现。

## 硬件平台

- **平台**: Xilinx FPGA（MicroBlaze v11.0, standalone）
- **加速器**: 4 个 NPU 模块 — GEMM、ALU、PixelShuffle、SESR Image
- **框架**: 精简版 ncnn 推理引擎

## 构建与运行

1. 在 Vitis 中通过 `sesr_accel.prj` 打开工程
2. 构建（`Release` 配置），将 ELF 下载到 MicroBlaze
3. 程序自动运行推理，通过 UART 输出性能计数器

## 导出输出图像

程序通过 UART 打印输出图像地址 <addr> 。使用 `mrd` 命令 dump 内存：

```
mrd -bin -file output.txt <addr> 921600
```

转换为 PNG：

```bash
python3 rgb2png.py output.txt
```

## 配置项

`src/config.h` 中的关键宏：

### 推理模式

| 宏 | 默认值 | 说明 |
|---|---|---|
| `SESR_SCALE` | `4` | 缩放因子：`2` → 640×360→1280×720，`4` → 320×180→1280×720 |
| `SESR_FORWARD_ON_NPU` | 开启 | 主开关——所有层在 NPU 上运行 |
| `SESR_FORWARD_ON_CPU` | 关闭 | 强制全 CPU 推理（覆盖 NPU 开关） |
| `SESR_FUSED_TAIL_STREAM_ADD_ON_NPU` | 开启 | 尾部融合后处理（PixelShuffle + Clip + YCbCr→BGR）在 NPU 执行 |
| `SESR_E2E_COMMAND` | 开启 | 端到端命令模式 |
| `SESR_FIXED_COMMAND_TABLE` | 开启 | 固定命令表（预编译，减少运行时开销） |

### 硬件参数

| 宏 | 值 | 说明 |
|---|---|---|
| `IFM_BUFFER_WIDTH` | 256 | 输入特征图 buffer 宽度 |
| `IFM_BUFFER_DEPTH` | 16384 | 输入特征图 buffer 深度 |
| `WGT_BUFFER_DEPTH` | 4608 | 权重 buffer 深度 |
| `AXI_DATA_WIDTH` | 128 | AXI 总线位宽 |
| `DMA_DATA_WIDTH` | 128 | DMA 数据位宽 |
| `HW_FREQ_MHZ` | 500 | 加速器时钟频率 |

### 调试

| 宏 | 默认值 | 说明 |
|---|---|---|
| `SESR_VERBOSE_LOG` | 关闭 | 打印每层名称和 shape |
| `NCNN_TOOLS` | 关闭 | 模拟模式，强制全 CPU 路径 |

## 目录结构

```
src/
├── main.cpp          入口
├── config.h          编译时配置
├── hw/               NPU 硬件驱动与中断
├── layer/            自定义 ncnn 层（卷积、PixelShuffle 等）
├── ncnn/             精简版 ncnn 推理引擎
├── net/              SESR 网络、命令表、色彩转换
├── models/           模型权重（M5/M11, ×2/×4）
└── utils/            工具函数
```
