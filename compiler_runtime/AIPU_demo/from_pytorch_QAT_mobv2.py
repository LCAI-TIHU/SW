import numpy as np
from PIL import Image
import torch
import torchvision

import tvm
from tvm import relay


model = torch.jit.load('/workspace/pytorchmodel/test.pth')
model = model.eval()
input_shape = [1, 3, 224, 224]
input_data = torch.randn(input_shape)
scripted_model = torch.jit.trace(model, input_data).eval()
print(scripted_model)

img_path ="/workspace/testimage/ILSVRC2012_val_00000293.JPEG"
img = Image.open(img_path)

my_preprocess = torchvision.transforms.Compose(
    [
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)
img = my_preprocess(img)
img = np.expand_dims(img, 0)


input_name = "x"
print("----------------",img.shape)
shape_list = [(input_name, img.shape)]
mod, params = relay.frontend.from_pytorch(scripted_model, shape_list)



