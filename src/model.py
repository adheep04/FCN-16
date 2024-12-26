import torch
from torch import nn
from torchvision.models import vgg16, VGG16_Weights

class FCN_16(nn.Module):
    '''
    class for a fully convolutional network described in the paper "Fully Convolutional Networks for Semantic Segmentation"
    uses part of a pretrained base model and includes upsampling using deconvolution
    
    
    '''
    def __init__(self, efficient=False):
        super().__init__()
        # pretrained feature extractor
        self.base = self._get_base(efficient)
    
    def _get_base(self, efficient):
        # get the vgg-16 pretrained model used in paper
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        
        # remove final classifier layers
        model.classifier = nn.Identity()
        model.avgpool = nn.Identity()
        
        
        
        if efficient:
            
        
        
        
        
        
