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
    transforms.ConvertImageDtype(torch.float32),    
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                       std=[0.229, 0.224, 0.225])
])

# paths
train = Path('./data/mapillary-vista/training/images')
labels = Path('./data/mapillary-vista/validation/v2.0/labels')

# get model
fcn8 = FCN(n_class=30, net='8')
fcn8 = fcn8.cuda()

paths = list(val.iterdir())
random_paths = sample(paths, k=100)  # k=number you want

try:
    print('start')
    for i, img_path in enumerate(random_paths): 
        print('step:', i,'_________________')

        t0 = time.time()
        img = Image.open(img_path)

        t1 = time.time()
        data = process(img=img).unsqueeze(0)
        data = data.cuda()
        
        print(f"Data shape: {data.shape}")

        t2 = time.time()
        pred = fcn8(data)
        
        print(f"Data shape: {pred.shape}")
        print(f"Forward time: {time.time()-t2:.2f}s")
except:
    print('____NORM MEANS____')
    

print('____NORM MEANS____')
print('p4 norms', fcn8.p4_mean)
print('score norms', fcn8.score_mean)
print('p3 norms', fcn8.p3_mean)
    
    
    

def to_tensor(img):
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    return transform(img)

cat_img = Image.open('./data/custom/cat108.jpg')
lion_img = Image.open('./data/custom/lion.jpg')

cat_tensor = to_tensor(cat_img)[:3,:,:].unsqueeze(0)
lion_tensor = to_tensor(lion_img)[:3,:,:].unsqueeze(0)
batch = torch.cat([cat_tensor, cat_tensor], dim=0)
print(batch.shape)

fcn = model.FCN(n_class=30, net='8')
# for name, module in fcn.base_net.classifier.named_children():
#     if isinstance(module, (nn.Conv2d)):
#         print(name, module)

# fcn.register_hooks()
pred = fcn(batch)




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



