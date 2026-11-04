# 主机端算法与补洞网络

## 内容

- `sgm/sgm.py`：原始 SGM 算法，与项目 SGM/SGM.py 一致。
- `depth_completion/`：原始网络全部 15 个 Python 源文件，包括训练、评估、数据加载、损失及网络定义。
- `depth_completion/workspace/exp_msg_matter/params.json`：模型配置。
- `depth_completion/workspace/exp_msg_matter/checkpoints/DataParallel_ep0036.pth.tar`：本次使用的 epoch 36 权重。
- `depth_completion/LICENSE`：上游许可证，源码内版权信息保留。

## 运行前准备

SGM 使用 NumPy、OpenCV、Numba、SciPy；补洞网络还使用 PyTorch、torchvision、Pillow、matplotlib 和 torchsummary。请根据 CPU/CUDA 环境安装匹配依赖，原训练环境的完整版本锁定文件未随原项目提供。

原脚本及 params.json 含原环境的数据集路径，运行前须改为实际 KITTI/Matterport 数据目录。完整训练数据集不包含在提交包内。应先阅读脚本的数据预处理及掩码约定，不能把 KITTI 和 Matterport 的样本组织直接混用。

补洞网络入口位于 depth_completion，评估入口为 run_eval.py，默认 epoch 36。主机算法入口与 VU440 固定样本部署是两个阶段；从新图像重新导出板端嵌入资源的完整工具链尚未闭合，不能声称任意输入一键上板。

固定样本无需重新训练即可离线验证，见 ../docs/reproduce.md；实板数据见 ../evidence/run1。
