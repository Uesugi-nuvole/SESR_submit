# SGM 视差补全系统

左右图像在电脑端执行 SGM；VU440 上的 MicroBlaze 与 NPU 协同执行补洞网络，最终视差存入 DDR，由调试接口读回。提交包以 V7 固定样本部署及真实板端结果为核心。

## 目录与入口
|目录|内容|
|---|---|
|deployment/v7|部署源码、固定输入、权重、软件参考、XSA/bit/MMI 和构建入口|
|deployment/run1|实际运行的 ELF 与 map|
|deployment/current_source_snapshot|当前 IDE 源码快照；不承诺与 run1 ELF 构建瞬间完全一致|
|host|SGM 与补洞网络的软件代码、模型权重|
|hardware/rtl|硬件 RTL；完整 Vivado 工程另外提供|
|software/super_resolution|超分平台驱动及软件源代码|
|skill|通用张量比较脚本、软件示例与真实板端输出|
|evidence/run1|DDR 原始结果、结果图、严格比较与 GT 评价|
|tools|文件完整性校验和固定样本 GT 评价|
|licenses|第三方许可证，另保留各源码目录内许可证|

先阅读 docs/reproduce.md。完整 Vivado 工程约 1.33 GB，不包含在比赛代码压缩包内，通过 GitHub Releases 附件提供：[下载完整 Vivado 工程](https://github.com/Uesugi-nuvole/SESR_submit/releases/download/submission-2026/sesr_accel_vivado.zip)，校验值见 docs/vivado_download.md。

本次证据为一个 128×128 裁剪样本，不代表整幅图实时处理或全数据集评测。板端完成运行并改善原 SGM 的 GT 指标，但与软件部署参考尚未逐元素对齐。
