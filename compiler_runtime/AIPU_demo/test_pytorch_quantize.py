import os
import torch
import torchvision.models.quantization as torch_qua

model =  torch_qua.mobilenetv2.mobilenet_v2(torch_qua.QuantizableMobileNetV2, quantize=True)

# torch.jit.save(torch.jit.script(model), 'test.pth')

_dummy_input_data = torch.rand(1, 3, 224, 224)
onnx_file = os.path.join('/work/test_torch_quantize.onnx')
torch.onnx.export(model, (_dummy_input_data,), onnx_file, verbose=True, opset_version=11)