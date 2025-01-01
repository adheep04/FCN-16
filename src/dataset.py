import torch
from torch.utils.data import Dataset
import pathlib
import math
from torchvision import transforms
from config import config
from PIL import Image
import numpy as np
 

class CityscapesDataset(Dataset):
    '''
    dataset for handling the cityscapes dataset. city folders have been combined to one (weimar, zurich, etc -> images)
    
    args:
    - data_path: Path()
        - folder of images (concatenated all images into one folder)
    - label_path: Path()
        - folder of labels (same as above)
    
    '''
    
    def __init__(self, data_path, label_path, transform=False): 
        
        # get dict {1: img_path, label_path, 2: ....}
        self.data_dict = self._get_dict(data_path, label_path)
        
        self.process_img = transform if transform else transforms.Compose([
            # convert to tensor (and scale [0,1])
            transforms.ToTensor(), 
            # normalize according to VGG preprocessing standards  
            # note: these are the mean and std values for the RBG channels in the original imagenet dataset the vgg was trained on 
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406], 
                std=[0.229, 0.224, 0.225]
                )
            ])
        
        
    def __getitem__(self, index):
        
        # get sample paths
        img_path = self.data_dict[index][0]
        label_path = self.data_dict[index][1]
        
        # get images
        data_img = Image.open(img_path)
        label_img = Image.open(label_path)
        
        
        # apply respective adjustments then resize image to save memory 
        data_tensor = self._resize(self.process_img(data_img))   
        label_tensor = self._resize(self.process_label(label_img))
        
        # map id to class labels for training
        self._map_labels(label_tensor)
        
        # validate sizes
        assert data_tensor.shape[1:] == label_tensor.shape[1:]
        assert torch.max(label_tensor) <= 20

        return data_tensor, label_tensor.squeeze(0).long()
        
        
    def __len__(self):
        return len(self.data_dict)
    
    
    def process_label(self, label_img):
        # convert image to tensor
        label_tensor = torch.from_numpy(np.array(label_img)).unsqueeze(0)
    
        
        
        assert len(label_tensor.shape) == 3
        
        return label_tensor
    
    def _map_labels(self, label_tensor, map=config.LABEL_MAP):
         # replace all 255/-1 values in label with -> 19 (ignore class)
        # uses bitwise or
        
        label_tensor[(label_tensor == 255) | (label_tensor == -1)] = 19
        # replace id with train class given map
        for id, train_id in map.items():
            label_tensor[label_tensor == id] = train_id
        
        
        
    def _resize(self, img, max_pixels=config.MAX_PIXELS):
        # max is 1024*1024 pixels
        
        # get original image dimensions
        w, h = img.shape[-2:]
        # calculate scale ratio between original image necessary to hit max pixel size
        ratio = math.sqrt(max_pixels / (w * h))
        new_w = int(w * ratio)
        new_h = int(h * ratio)
        # resize height and width by proportional values to maintain aspect ratio
        return transforms.Resize((new_h, new_w))(img)
    
    
    def _get_dict(self, img_dir, label_dir):
        data_dict = {}
        
        # iterate through img folder and map indices to img/label samples: 
        for i, img_path in enumerate(img_dir.iterdir()): 
            # 'leftImg8bit' is a substring in the name of the img files
            id = img_path.stem.replace("_leftImg8bit", "")  
            # 'gtFine_color' is a substring in the name of label files
            label_path = label_dir / f"{id}_gtFine_labelIds.png"
            data_dict[i] = img_path, label_path
        return data_dict