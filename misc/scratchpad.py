from model import FCN
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn
from pathlib import Path
import numpy as np
import time
from random import sample
import math


def resize(img, max_pixels=1024*1024):
    w, h = img.size
    ratio = math.sqrt(max_pixels / (w * h))
    new_w = int(w * ratio)
    new_h = int(h * ratio)
    return transforms.Resize((new_h, new_w))(img)

process = transforms.Compose([
    transforms.Lambda(lambd=resize),
    transforms.PILToTensor(),  
])

# paths

# # get model
# fcn8 = FCN(n_class=30, net='8')
# fcn8 = fcn8.cuda()

data = Path('./data/cityscapes/train/img')
label = Path('./data/cityscapes/train/label')

data_dict = {}
for i, data_path in enumerate(data.iterdir()): 
    id = data_path.stem.replace("_leftImg8bit", "")  
    label_path = label / f"{id}_gtFine_labelIds.png"
    data_dict[i] = data_path, label_path
    
    
lable = Image.open(data_dict[2][1])
tensor = transforms.PILToTensor()(lable)


assert torch.max(tensor) < 32
tensor[(tensor == 255) | (tensor == -1)] = 19
tensor[tensor == 33] = 19

print(torch.max(tensor))


    
    
    
    
print('start')
count = 0
for i, path in enumerate(data.iterdir()): 
    id = path.stem.replace("_leftImg8bit", "")  
    label_path = label / f"{id}_gtFine_color.png"
    if not label_path.exists():
        print(label_path)
        print('error')
        count += 1

print(count)
    

# for item in label.iterdir():
#     print(item) 
    
    

# def to_tensor(img):
#     transform = transforms.Compose([
#         transforms.ToTensor()
#     ])
#     return transform(img)

# cat_img = Image.open('./data/custom/cat108.jpg')
# lion_img = Image.open('./data/custom/lion.jpg')

# cat_tensor = to_tensor(cat_img)[:3,:,:].unsqueeze(0)
# lion_tensor = to_tensor(lion_img)[:3,:,:].unsqueeze(0)
# batch = torch.cat([cat_tensor, cat_tensor], dim=0)
# print(batch.shape)

# fcn = model.FCN(n_class=30, net='8')
# for name, module in fcn.base_net.classifier.named_children():
#     if isinstance(module, (nn.Conv2d)):
#         print(name, module)

# fcn.register_hooks()
# pred = fcn(batch)




''' hooks for monitering'''

    # def register_hooks(self):
    #     def dimension_hook(module, input, output):
    #         # Get the module's name or class if name not available 
    #         name = module.__class__.__name__
    #         if hasattr(module, 'original_name'):
    #             name = module.original_name
                
    #         # Format input/output shapes
    #         input_shape = input[0].shape if isinstance(input, tuple) else input.shape
    #         output_shape = output.shape if hasattr(output, 'shape') else None
            
    #         # Print with clear formatting
    #         print(f"Output shape: {output_shape}")
    #     def _register_on_modules(module, prefix=''):
    #         for name, child in module.named_children():
    #             # Store original name for better debugging
    #             child.original_name = f"{prefix}.{name}" if prefix else name
    #             child.register_forward_hook(dimension_hook)
    #             _register_on_modules(child, child.original_name)
                
    #     _register_on_modules(self)



# def print_shape(name):
#     def hook(module, input, output):
#         print(f"{name}: {output.shape}")
#     return hook

# # Register hooks
# for name, layer in fcn.base_net.features.named_children():
#     layer.register_forward_hook(print_shape(f"Layer {name}"))

# for name, layer in fcn.base_net.classifier.named_children():
#     layer.register_forward_hook(print_shape(f"Layer {name}"))

# # for name, layer in fcn.classifier.named_children():
# #     layer.register_forward_hook(print_shape(f"Layer {name}"))



