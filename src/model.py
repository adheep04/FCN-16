from torch import nn
import torch
from torchvision.models import vgg16, VGG16_Weights
import torchvision.transforms.functional as F

class FCN(nn.Module):
    '''
    class for a fully convolutional network described in the paper "Fully Convolutional Networks for Semantic Segmentation"
    uses an adapted version of a pretrained base model, includes upsampling using deconvolution/bilinear interpolation, and uses 
    skip connections from shallower layers.
    
    args: 
    - n_class: int
        - number of classes to identify
    - net: str in ['32', '16', '8']
        - which fcn variant this is
    '''
    
    def __init__(self, n_class, net='32'):        
        super().__init__()
        
        # validate fcn variant input
        assert net in ['32', '16', '8']
        
        self.net = net
        
        # pretrained feature extractor
        self.base_net = VGG16()
        
        # upsamples pool5 pred to fuse with pool4 pred in fcn16 and fcn8
        self.upsample2_1 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4,stride=2, bias=False)
       
        # upsamples the fuse result of pool4 and pool5 pred to fuse with pool3 pred in fcn8
        self.upsample2_2 = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4,stride=2, bias=False)
        
        # final upsample is essentially a bilinear interpolation and can be used by either net given the original spatial dim of image
        self.score_final = FixedInterpolation()
        
        
        # replace last 3 fully connected with convolution layers
        self._init_helper(n_class)
    
    def _init_helper(self, n_class):
        '''
        replaces the last 3 linear layers with convolution layers
        maintaining trained parameter values. 
        
        adds 1x1 conv layer after pool4 for future image reconstruction
        
        (my reimplementation of surgery.transplant() :P)
        '''
        
        # load pretrained vgg16 model to do "surgery"
        self.base_net.load_state_dict(vgg16(weights=VGG16_Weights.DEFAULT).state_dict())



        ''' replace last 3 linear layers with convolutions'''

        # get weights and biases from last 2 fully connected layers in VGG-16
        fc6_weights = self.base_net.classifier[0].weight.data.reshape(4096, 512, 7, 7)
        fc7_weights = self.base_net.classifier[3].weight.data.reshape(4096, 4096, 1, 1)
        fc6_bias = self.base_net.classifier[0].bias.data
        fc7_bias = self.base_net.classifier[3].bias.data
        
        # use pytorch's convolution layer class to initialize conv layers
        fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        pred = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1)
        
        # replace default parameters with trained reshaped weights (except prediction layer)
        fc6.weight.data = fc6_weights
        fc7.weight.data = fc7_weights
        pred.weight.data = torch.zeros(n_class, 4096, 1, 1)
        fc6.bias.data = fc6_bias
        fc7.bias.data = fc7_bias

        # replace linear layers with convlayers
        self.base_net.classifier[0] = fc6
        self.base_net.classifier[3] = fc7
        self.base_net.classifier[6] = pred
        
        
        
        ''' other base model adjustments'''
        
        # remove average pooling layer from vgg
        self.base_net.avgpool = nn.Identity()
        
        # initialize post-pool skip connection layers to store intermediate values
        # insert skip connection prediction layers in sequential after pool3 and pool4
        self.base_net.features.insert(17, SkipConnection(in_channels=256, n_class=n_class)) # pool3 pred is shape (b, c, h//8, w//8)
        self.base_net.features.insert(25, SkipConnection(in_channels=512, n_class=n_class)) # pool4 pred is shape (b, c, h//16, w//16)
        
        # initialize deconvolution weights like that of bilinear interpolation and have it learnable
        self._bilinear_weight_init(self.upsample2_1)
        self._bilinear_weight_init(self.upsample2_2)
        
        # replace fc6 padding from (1, 1) -> (3, 3)
        ''' note: this detail isn't part of the original implementation'''
        self.base_net.classifier[0].padding = (3, 3)

    def forward(self, x):
        '''
        forward pass in end-to-end image segmentation model FCN
        
        progression of function models the progression of fcn variants and the development of the prediction.
        
        args:
        - x: tensor(batch_size, channel_size, height, width)
        
        output:
        - tensor(batch_size, class_size, height, width)
        
        '''
        
        
        '''fcn 32'''
        # get img spatial dimensions
        img_spatial_dims = (x.shape[-2], x.shape[-1])
        
        # pass through base vgg16 to get coarse class predictions
        pool5_pred_f = self.base_net.features(x)
        pool5_pred = self.base_net.classifier(pool5_pred_f)
        
        # if net is fcn32, upsample back to original spatial dimensions and return
        if self.net=='32': return self.score_final(pool5_pred, out_dim=img_spatial_dims)
        
        
        '''fcn 16'''
        # upsample by 2
        pool5_upsampled = self.upsample2_1(pool5_pred)
        
        # get stored pool4 pred for net 16 and 8
        pool4_pred = self.base_net.features[25].val
        
        # crop upsampled score to align with skip connection dimensions
        pool5_cropped = self._crop(big=pool5_upsampled, small=pool4_pred)
    
        # fuse both (sum them)
        fuse1 = pool5_cropped + pool4_pred
        
        # bilinear upsample back to image spatial dim and return
        if self.net=='16': return self.upsample_final(fuse1, out_dim=img_spatial_dims)
        
        
        '''fcn 8'''
        # upsample by 2
        fuse1_upsampled = self.upsample2_2(fuse1)
        
        # get pool3 pred for net 8
        pool3_pred = self.base_net.features[17].val
        
        # crop upsampled score to align with skip connection dimensions
        fuse1_cropped = self._crop(big=fuse1_upsampled, small=pool3_pred)
        
        # fuse pool3 pred and upsampled fuse1
        fuse2 = fuse1_cropped + pool3_pred
        
        # bilinear upsample and return
        if self.net=='8': return self.score_final(fuse2, out_dim=img_spatial_dims)
        
        
    def _crop(self, big, small):
        '''
        crops big to be the same dimension as small given
        that they are two tensors and the last two dimensions of small
        is smaller than that of big
        
        '''
        
        # get dimensions
        h_small, w_small = small.shape[2:]
        h_big, w_big = big.shape[2:]
        
        assert h_big > h_small and w_big > w_small, f"upsampled {big.shape} spatial dimensions doesn't extend pool_score's {small.shape}"

        return F.center_crop(img=big, output_size=(h_small, w_small))
        
    def _bilinear_weight_init(self, conv):
        pass
                
    def register_hooks(self):
        def dimension_hook(module, input, output):
            # Get the module's name or class if name not available 
            name = module.__class__.__name__
            if hasattr(module, 'original_name'):
                name = module.original_name
                
            # Format input/output shapes
            input_shape = input[0].shape if isinstance(input, tuple) else input.shape
            output_shape = output.shape if hasattr(output, 'shape') else None
            
            # Print with clear formatting
            print(f"Output shape: {output_shape}")
        def _register_on_modules(module, prefix=''):
            for name, child in module.named_children():
                # Store original name for better debugging
                child.original_name = f"{prefix}.{name}" if prefix else name
                child.register_forward_hook(dimension_hook)
                _register_on_modules(child, child.original_name)
                
        _register_on_modules(self)
    
    

class FixedInterpolation(nn.Module):
    '''
    final upsample to the input dimensions in a FCN
    
    '''
    def forward(self, x, out_dim):
        # bilinear fixed interpolation to "increase the resolution"
        return nn.functional.interpolate(input=x, size=out_dim, mode='bilinear') 
          

class SkipConnection(nn.Module):
    '''
    applies a convolution, upsamples 2x, and stores the resulting output,
    but passes its input forward untouched to the next layer

    '''
    def __init__(self, in_channels, n_class):
        super().__init__()
        # pool4 has 512 output channels
        # 30 output channels for each class
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=n_class, kernel_size=1, stride=1)
        # upsample layer
        self.val = None
        # initialize weights with 0
        nn.init.zeros_(self.conv.weight) 

    def forward(self, x):
        '''
        returns the input as is but stores the result of a convolution.
        
        args
        - x: tensor(batch_size, 512, img_height/16, img_width/16)
        
        '''
        # apply convolution, upsample 2x, and store
        self.val = self.conv(x)
        # apply nothing to x
        return x
    
class VGG16(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.features = nn.Sequential(
            
            # conv 1
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=3, stride=1, padding=(1, 1)),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # conv 2
            nn.Conv2d(64, 128, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(128, 128, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # conv 3
            nn.Conv2d(128, 256, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # conv 4
            nn.Conv2d(256, 512, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            
            # conv5
            nn.Conv2d(512, 512, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, (1, 1)),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool2d((7, 7))
        
        self.classifier = nn.Sequential(
            
            # these layers will be replaced by convolutions
            nn.Linear(in_features=25088, out_features=4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=4096, bias=True),
            nn.ReLU(True),
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(in_features=4096, out_features=1000, bias=True),
        )
        