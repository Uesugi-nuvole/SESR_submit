# SESR Submit

SGM 视差补全 FPGA 竞赛提交材料。电脑端运行 SGM，VU440 上的 MicroBlaze 与 NPU 协同执行 V7 补洞网络，结果从 DDR 回读后评价。

## 阅读与使用
- 代码入口：[code/README.md](code/README.md)，运行方法：[code/docs/reproduce.md](code/docs/reproduce.md)。离线命令须先进入 code 目录执行。
- 技术报告：[reports/SESR_Technical_Report.pdf](reports/SESR_Technical_Report.pdf)，可编辑 Word 同目录保存。
- 实现后硬件截图：hardware/reports；板端原始输出及评价：code/evidence/run1。
- 验证工具：code/skill；固定样本、权重与软件参考：code/deployment/v7。
- 比赛代码附件：submission/SESR_Code.rar；演示视频：media/video。
- 完整 Vivado 工程：release_assets/sesr_accel_vivado.zip，计划作为 GitHub Releases 独立附件。

## 本地材料与 GitHub
本目录为统一提交入口，原始工程未移动或覆盖。普通 Git 文件保留代码、说明、报告与必要证据。完整 Vivado、比赛 RAR 和视频仅保留在本地并计划作为 Releases 附件发布。大于 100 MiB 的 bit 位于比赛 RAR 内，不放普通 Git 历史。

GitHub 仓库：https://github.com/Uesugi-nuvole/SESR_submit 。已创建并关联 origin，目前为私有空仓库，尚未推送文件或发布 Releases。报告中的实际附件下载链接需在发布后补齐再提交。

固定样本结果并非整幅图实时性能：已有一次 128×128 板端输出，板端与软件部署参考尚未逐元素对齐。详见 code/docs/validation.md。

第三方许可证保留在 code/licenses 及相应源码目录。签字表、证件及其他个人材料仅放 private_local，禁止提交 Git。
