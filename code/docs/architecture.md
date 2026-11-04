# 系统架构与数据接口

## 应用数据流
左右图像 → 主机 SGM → 待补全视差、RGB、有效掩码 → 网络 → 输出视差 → DDR 回读 → 与部署参考和 GT 分别比较。

## 板端分工
MicroBlaze 执行网络调度、软件算子及后缩放；NPU 执行普通卷积的整数累加。DDR 存放程序、张量与输出。AXI 连接控制与数据接口，中断通知完成，UART 输出运行日志。硬件模块存在不表示本次网络调用了全部模块。

## 固定样本入口
样本为 000082_10，原图裁剪 x448:576、y128:256。input.npz 中 input_disp 为128×128，rgb为128×128×3，mask为128×128；mask=1表示原输入有效，mask=0表示空洞。板端资源已编入ELF，本包不宣称已实现任意新图的实时上传链路。

最终节点143输出1×128×128 float32，小端序，共65536字节。节点144不是最终输出。运行后输出地址以DONE日志为准，不固化为通用地址。

## 代码入口
SGM：host/sgm/sgm.py 的 SGM 函数。其原始 main 仍含数据集绝对路径，批量数据集运行需要配置。
模型：host/depth_completion/workspace/exp_msg_matter/network_exp_msg_matter.py。
板端：deployment/v7/source/sesr_accel/src/main.cpp 与 completion/completion_graph_runtime.h。
硬件：hardware/vivado 中 system.bd、top.xdc 与 tcp.v；hardware/development 提供相应研发源码。

## 两类验证
部署一致性：板端对整数软件参考，本次未通过严格容差。
应用质量：同一GT有效区域上比较SGM、软件补洞、板端补洞，并拆分原空洞/已知区域。本次单裁剪板端MAE约0.946px，D1式坏点率约3.68%；不能当作KITTI全数据集成绩。约110分钟是用户观察耗时，未作为可靠性能基准。
