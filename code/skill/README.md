# FPGA 部署结果一致性验证

适用于已经取得软件参考和设备输出的项目。两份文件表示同一输出节点、同一样本的连续 C 顺序张量。可在没有 FPGA 或 Vitis 的电脑执行。需要 Python 3.8+ 和 NumPy（python -m pip install numpy）。以下命令在 skill 目录执行。

## 先运行随包案例
    python scripts/compare.py examples/vu440/host_daylight_final.bin examples/vu440/deployment_postscale_final_le_f32.bin --shape 1 1 128 128 --dtype float32 --actual-endian little --reference-endian little --atol 1e-5 --rtol 1e-4 --report host_check.json

样本为 KITTI 000082_10 的 128×128 裁剪，input.npz 保存 RGB、带洞视差、掩码和仅用于评价的 GT。输入文件用于固定实验身份，无需传给比较器。host_daylight_final.bin 来自主机静态图执行，参考来自独立整数软件计算。实测得到 16384 个有限值、MAE 0、超差 0；随包 host_report.json 可核对。

## 比较自己的板端结果
    python scripts/compare.py board_output.bin software_ref.bin --shape 1 1 128 128 --dtype float32 --actual-endian little --reference-endian little --atol 1e-5 --rtol 1e-4 --report board_report.json

将两个路径、形状、类型、字节序和容差替换为自己模型的输出约定。对整数码的精确比较可省略两个容差（默认均为零）。支持 float16/32 及 8/16/32 位有符号和无符号整数。相同类型、元素位置对应且数值单位一致时可比较。量化尺度或零点不同，应先按各自参数反量化到相同单位。每通道量化须按正确通道轴广播。

输入为无头 raw binary。每个文件的长度必须等于各维乘积乘元素字节数。若源数据是 npy/npz，先确认键、dtype、轴序、形状并导出连续数组；图片、带头文件、DMA padding、分块或通道重排数据不能当成连续张量直接比较。原始文件自身不能证明格式、布局或字节序，长度正确也不能自动排除解释错误。

输出为终端摘要及指定 JSON（父目录须存在），包含 SHA256、输入格式、有限性、平均/最大绝对差、首个超差坐标和超差数。逐元素判定 abs(actual-ref) <= atol + rtol*abs(ref)。非有限值计为超差，全局误差统计为 null。退出码 0 表示通过，1 表示数值不符，2 表示参数或文件异常。容差应在实验前固定。脚本在内存中比较单个输出，多输出模型逐节点运行。

## 实板案例材料与经验
examples/vu440/board_output.png 与 board_metrics.png 是上板组回传的真实视觉结果和 GT 评估。板端对 GT 的 MAE 为 0.946 像素。原空洞且有 GT 的 451 点为 0.975 像素。本提交包已补入 board_output_run1.bin 和 board_report.json。按 atol=1e-5、rtol=1e-4 比较，MAE 为 0.6854828118，最大绝对差为 8.900672913，超差 16333/16384；数值尚未严格对齐。不要把软件输出复制成板端样例。

经验来自本项目分层验证：主机与整数参考一致，而原 PyTorch 在量化阈值附近一个码的改变会传到后层。遇到超差，先核对样本、节点、形状、字节数和字节序，再看布局和尺度；仍不一致则找首个分歧节点，用相同输入重算该算子。板端与软件各自对 GT 的 MAE 之差不能代替逐元素比较。在改变板子、模型和输出尺寸时都可复用这个流程。
