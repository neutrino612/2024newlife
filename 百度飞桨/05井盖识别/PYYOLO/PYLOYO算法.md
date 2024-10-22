PP-YOLOE+ 单阶段Anchor-free目标检测模型

## 1. 模型简介

PP-YOLOE+ 是卓越的单阶段 Anchor-free 目标检测模型，超越了多种流行的 YOLO 模型。PP-YOLOE+ 避免了使用诸如 Deformable Convolution 或者 Matrix NMS 之类的特殊算子，以使其能轻松地部署在多种多样的硬件上。


## 2. 技术方案

PP-YOLOE+ 系列模型的主要改进点有：

1. 可扩展的 backbone 和 neck。
2. Task Alignment Learning。
3. Efficient Task-aligned head with DFL 和 VFL。
4. SiLU(Swish) 激活函数。

## 3. 如何使用

### 3.1 数据准备

在完成数据标注工作后，请按照数据集规范说明检查数据组织格式是否满足要求。如果不满足，请参照规范说明进行调整，否则将导致后面的数据校验环节无法通过。

完成数据准备后，点击右上角【创建模型产线】。

在 AI Studio 云端，可以通过挂载数据集的方式来使用自己的数据。模型产线创建和修改窗口可以挂载和管理数据集，如下图所示。数据集挂载成功后，需在开发者模式中查看并解压挂载的数据集，数据集路径查看方式为【开发者模式】-左侧【资源管理器】，在目录 `AISTUDIO/data` 下，数据集解压方式为【开发者模式】-左侧【菜单】-【终端】-【新建终端】，在【终端】中通过命令行方式解压，并将解压后的数据集路径拷贝到【工具箱模式】-【数据校验】-【数据集路径】。**注意：在 AI Studio 云端，`data` 目录不持久化存储，如需持久化存储，需要将数据解压至其它目录。**

在本地端-工具箱模式中，您可以直接在【工具箱模式】-【数据校验】-【数据集路径】中填写本地数据路径来使用自己的数据集。

在配置完成【数据集路径】后，即可点击【开始校验】按钮进行数据集校验。

https://aistudio.baidu.com/modelsdetail/33?modelId=33

https://aistudio.baidu.com/pipeline/step/p-388af27dd365  产线模型启动
