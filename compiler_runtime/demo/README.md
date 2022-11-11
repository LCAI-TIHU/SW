# TIHU DEMO
TIHU demo are test cases we provide to help users quickly get familiar with project components
> Please make sure that you have installed the required environment, otherwise please refer to the installation tutorial
# Preparing the models
TIHU currently supports the following models
| Model | Checkpoint | 
| :----: | :----:  |
|[LeNet](http://vision.stanford.edu/cs598_spring07/papers/Lecun98.pdf)|[lenet.pb](https://pan.baidu.com/s/1Dk_cYJM5wt854xtb4K7Hdw?pwd=TIHU)|
| [ResNet V1 50](https://arxiv.org/abs/1512.03385) |[resnet_v1_50_2016_08_28.tar.gz](http://download.tensorflow.org/models/resnet_v1_50_2016_08_28.tar.gz) |
| [MobileNet_v2_1.0_224](https://arxiv.org/abs/1801.04381) |[mobilenet_v2_1.0_224.tgz](https://storage.googleapis.com/mobilenet_v2/checkpoints/mobilenet_v2_1.0_224.tgz) |
# Evaluating performance of a quantized model
> Taking resnet50 for example
## Frozen model
Before that, the pre-trained model needs to be frozen to remove redundant parameters. We provide a script to freeze the model, you can customize the output node by setting output_name according to your needs.
```bash
$ python3 frozen_model.py
```
## Preparing the datasets
Using [ImageNet](https://image-net.org/challenges/LSVRC/2012/) validation dataset to verify the performance of a quantized model.
## Hyperparameter Description
- label_offset: Set the label offset, for the case where the first category of the model prediction result is the background, for example, the mobilenet prediction result are 1001 categorys , of which the first category is the background.
- calibration_samples: Set the calibration dataset size, the selection criteria should cover all categories.
- weight_scale: Set the weight quantization method; we add the implementation of per_channel based on the TVM implementation of per_tensor quantization, which can be used to compare the accuracy of aipu per_tensor and per_channel, using "channel_max" to achieve.
- target: Set different hardware backends. If it is set to "aipu" backend, you need to complete the FPGA board programming.
## Evaluating the quantitative model
```bash
$ python3 from_tensorflow_quantize_resnet50_v1.py 2>&1 | tee quantize_resnet50_v1.log
```
