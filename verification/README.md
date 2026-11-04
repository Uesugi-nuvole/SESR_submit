# 验证入口

为保留脚本的相对路径，验证代码与原始数据统一位于 code 中，不另复制第二份。

1. 进入 ../code，执行 python tools/verify_manifest.py。
2. 执行 python tools/evaluate_gt.py 复核同一 GT 有效区域上的精度。
3. 按 skill/README.md 比较软件参考与真实板端输出。

原始文件：../code/evidence/run1/board_output_run1.bin。
逐元素比较与 GT 应用指标分开评价。
