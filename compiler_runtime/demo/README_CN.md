# TIHU 示例
TIHU示例是我们提供的测试用例，旨在帮助使用者快速熟悉项目组件。
> 请确保您已安装项目运行所需环境，否则请查阅安装教程安装相应环境。
# 模型准备
TIHU目前支持模型如下：
| 模型 | 预训练 | 
| :----: | :----:  |
|[LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)|[lenet.pb](https://pan.baidu.com/s/1Dk_cYJM5wt854xtb4K7Hdw?pwd=TIHU)|
| [ResNet V1 50](https://arxiv.org/abs/1512.03385) |[resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) |
| [MobileNet_v2_1.0_224](https://arxiv.org/abs/1801.04381) |[mobilenet_v2_1.0_224.tgz](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) |
# 运行示例
> 以resnet50为例
## 冻结模型
在运行示例之前，需要将预训练模型冻结，祛除冗余参数。我们提供了冻结模型的脚本，您可以根据自己的需求，通过设置output_name自定义输出节点。
```bash
$ python3 frozen_model.py
```
## 数据准备
测试数据使用[ImageNet](https://image-net.org/challenges/LSVRC/2012/)的验证集，以验证模型经过量化后精度的变化。
## 超参数说明
- label_offset：设置标签偏移量，针对模型预测结果第一类为背景的情况，如mobilenet预测结果为1001类，其中第一类为背景类。
- calibration_samples：设置校准数据集大小，选取标准应囊括所有类别。
- weight_scale：设置权重量化方式；我们在TVM实现per_tensor量化的基础上增加了per_channel的实现，可以用来对比aipu per_tensor和per_channel的精度；使用”channel_max“实现。
- target： 设置不同的硬件后端，若设置为“aipu”后端，需完成FPGA板卡烧写。
## 运行示例
```bash
$ python3 from_tensorflow_quantize_resnet50_v1.py 2>&1 | tee quantize_resnet50_v1.log
```
