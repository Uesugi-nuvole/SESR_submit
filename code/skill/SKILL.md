---
name: fpga-output-check
description: Compare dense raw FPGA inference readbacks with a software deployment reference and report format and numerical discrepancies.
---

核对同一样本、模型版本、输出节点、布局及量化单位。按 README.md 中的命令调用 scripts/compare.py，保留两份输入的来源、哈希、比较容差和 JSON。先对部署参考做逐元素比较，再独立讨论应用精度。缺少真实回读时，明确区分软件回归与板端证据。
