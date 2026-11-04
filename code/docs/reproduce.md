# 复现与运行说明

## 离线结果验证
Python需要numpy与matplotlib。在包根目录执行：
```powershell
python tools/verify_manifest.py
python deployment/v7/compare_board_output.py evidence/run1/board_output_run1.bin deployment/v7/reference/deployment_postscale_final_le_f32.bin --dump-endian little --report comparison_local.json
python deployment/v7/render_output.py evidence/run1/board_output_run1.bin --dump-endian little --out comparison_local.png
python tools/evaluate_gt.py
```
compare脚本本次返回非零是已知的数值不一致，不是执行步骤缺失。不要放宽容差凑通过。

## 板端程序构建
当前采用Vitis2022.2、MicroBlaze，已知板卡器件xcvu440-flga2892-2-e。平台使用deployment/v7/original_platform/system_wrapper.xsa。包内generate_platform.tcl接受新workspace和XSA两个参数，生成BSP；rebuild.py接受--compiler、--bsp、--out，原有v7源码路径布局已保留。
这条构建入口在新目录尚未实际重编；不要宣称生成ELF与run1逐字节一致。run1测量对应的是单独保留的Release ELF。当前IDE源码也单独保留，便于后续比对。

## 结果回读
打开XSCT、连接实验室实际hw_server并选择正确MicroBlaze。等待本次DONE和STATUS 0，执行stop后：
```tcl
mrd -size w -bin -file output.bin <本次DONE地址> 16384
file size output.bin
```
长度必须为65536字节。首次保存成功前不复位、不重新Run、不烧写其他程序。

## 硬件工程
完整 Vivado 工程独立提供，见 vivado_download.md。代码包中的 original_platform 可用于软件平台构建和已有硬件配置；本次整理没有运行 Vivado，也未证明完整工程在全新环境重建通过。

## 算法训练与输入重导出
host代码保留原算法入口；完整KITTI/Matterport数据集不随包分发。修改数据路径需遵循各数据集授权。新样本生成嵌入资源、静态图与部署参考的完整导出脚本尚未确认到齐，不声称从任意左右图一键复现板端结果。
