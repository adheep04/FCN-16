import model
from torchvision import transforms
from PIL import Image
import torch
import torch.nn as nn



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

fcn.register_hooks()
pred = fcn(lion_tensor)






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



