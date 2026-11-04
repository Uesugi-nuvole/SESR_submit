# SESR Submit

SGM 视差补全 FPGA 竞赛提交材料。电脑端运行 SGM，VU440 上的 MicroBlaze 与 NPU 协同执行补洞网络，结果从 DDR 回读后评价。

## 提交文件与下载
|材料|位置|
|---|---|
|完整代码 RAR（含配套 bit）|[SESR_Code.rar](https://github.com/Uesugi-nuvole/SESR_submit/releases/download/submission-2026/SESR_Code.rar)|
|完整 Vivado 工程|[sesr_accel_vivado.zip](https://github.com/Uesugi-nuvole/SESR_submit/releases/download/submission-2026/sesr_accel_vivado.zip)|
|演示视频|[SESR_Demonstration.mp4](https://github.com/Uesugi-nuvole/SESR_submit/releases/download/submission-2026/SESR_Demonstration.mp4)|
|技术报告 PDF|[SESR_Technical_Report.pdf](reports/SESR_Technical_Report.pdf)|
|可编辑技术报告|[SESR_Technical_Report.docx](reports/SESR_Technical_Report.docx)|

[全部发布附件](https://github.com/Uesugi-nuvole/SESR_submit/releases/tag/submission-2026)。文件大小和 SHA256 见 submission/file_manifest.json。

## 代码与验证入口
- [代码说明](code/README.md)与[复现步骤](code/docs/reproduce.md)。离线命令先进入 code 目录执行。
- [SGM 与补洞网络](code/host/README.md)：包含 SGM.py 原始算法、补洞网络全部 15 个 Python 源文件及 epoch 36 权重；完整训练数据集不分发。
- 实现后硬件截图：hardware/reports；板端原始输出及评价：code/evidence/run1。
- 通用比较工具：code/skill；固定样本、权重、软件参考与构建入口：code/deployment/v7。
- 平台 bit 从完整代码 RAR 中获取；完整 Vivado 工程作为独立附件，不放普通 Git 历史。

## 实验范围
已有一次固定 128×128 样本板端输出。相同 GT 有效像素上的板端 MAE 为 0.946 px，原 SGM 为 2.211 px；板端与软件部署参考尚未逐元素对齐。这不是整幅图实时性能或官方 KITTI 排行榜结果，详见 code/docs/validation.md。

第三方许可证保留在 code/licenses 和对应源码目录。签字授权书及个人材料不公开提交。普通代码、报告和必要证据在仓库中管理，大型发布文件在 Releases 中分发。
