from torch import nn
import torch
from torchvision.models import vgg16, VGG16_Weights
from torchvision.transforms.functional import center_crop

class FCN(nn.Module):
    '''
    class for a fully convolutional network described in the paper "Fully Convolutional Networks for Semantic Segmentation"
    uses an adapted version of a pretrained base model, includes upsampling using deconvolution/bilinear interpolation, and uses 
    skip connections from shallower layers. Slightly modified in some places (like no intiial padding)
    
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
        
        # pretrained base net extractor
        self.base_net = VGG16()
        # upsamples pool5 pred to fuse with pool4 pred in fcn16 and fcn8
        self.upsample_a = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4,stride=2, bias=False)
        # upsamples the fuse result of pool4 and pool5 pred to fuse with pool3 pred in fcn8
        self.upsample_b = nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4,stride=2, bias=False)
        # final upsample is essentially a bilinear interpolation and can be used by either net given the original spatial dim of image
        self.upsample_final = FixedInterpolation()
       
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
        # original fcn paper used AlexNet and GoogleNet as well but found the best results using VGG16
        self.base_net.load_state_dict(vgg16(weights=VGG16_Weights.DEFAULT).state_dict())


        ''' replace last 3 linear layers with convolutions that have the same weights'''

        # get weights and biases from 2/3 fully connected layers in VGG-16
        # last prediction layer is replaced and initialized with zeros (ImageNet is tra)
        fc6_weights = self.base_net.classifier[0].weight.data.reshape(4096, 512, 7, 7)
        fc7_weights = self.base_net.classifier[3].weight.data.reshape(4096, 4096, 1, 1)
        fc6_bias = self.base_net.classifier[0].bias.data
        fc7_bias = self.base_net.classifier[3].bias.data
        
        # use pytorch's convolution layer class to initialize conv layers
        fc6 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        fc7 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        score = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1)
        
        # replace default parameters with trained reshaped weights (except prediction layer)
        fc6.weight.data = fc6_weights
        fc7.weight.data = fc7_weights
        score.weight.data = torch.zeros(n_class, 4096, 1, 1)
        fc6.bias.data = fc6_bias
        fc7.bias.data = fc7_bias

        # replace linear layers with convlayers
        self.base_net.classifier[0] = fc6
        self.base_net.classifier[3] = fc7
        self.base_net.classifier[6] = score
        
        
        ''' other base model adjustments'''
        
        # remove average pooling layer from vgg
        self.base_net.avgpool = nn.Identity()
        
        # initialize post-pool skip connection layers to store intermediate values
        # insert skip connection prediction layers in sequential after pool3 and pool4
        self.base_net.features.insert(17, SkipConnection(in_channels=256, n_class=n_class)) # pool3 score is shape (b, c, h//8, w//8)
        self.base_net.features.insert(25, SkipConnection(in_channels=512, n_class=n_class)) # pool4 score is shape (b, c, h//16, w//16)
        
        # initialize deconvolution weights like that of bilinear interpolation and have it learnable
        # https://github.com/tnarihi/caffe/commit/4f249a00a29432e0bb6723087ec64187e1506f0f <- used this code to produce the follow initialization
        bilinear_interp_weights = torch.tensor(
            [[[[0.0625, 0.1875, 0.1875, 0.0625],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.1875, 0.5625, 0.5625, 0.1875],
            [0.0625, 0.1875, 0.1875, 0.0625]]]]
        # duplicate along in_channel and out_channel dimension for size (30, 30, 4, 4)
        ).repeat((30, 30, 1, 1))
        
        self.upsample_a.weight.data = bilinear_interp_weights
        self.upsample_b.weight.data = bilinear_interp_weights
        
        # replace fc6 padding from (1, 1) -> (3, 3)
        ''' note: this detail isn't part of the original implementation'''
        self.base_net.features[0].padding = (8, 8)
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
        score_fr = self.base_net.features(x)
        score_fr = self.base_net.classifier(score_fr)
        
        # if net is fcn32, upsample back to original spatial dimensions and return
        if self.net=='32': return self.upsample_final(score_fr, out_dim=img_spatial_dims)
        
        
        '''fcn 16'''
        # upsample by 2
        score_fr_upsampled = self.upsample_a(score_fr)
        
        # get stored pool4 pred for net 16 and 8
        score_p4 = self.base_net.features[25].val
        
        # crop upsampled score to align with skip connection dimensions
        score_fr_cropped = self._crop(big=score_fr_upsampled, small=score_p4)

        # fuse both (sum them)
        fuse1 = score_fr_cropped + score_p4
        
        # bilinear upsample back to image spatial dim and return
        if self.net=='16': return self.upsample_final(fuse1, out_dim=img_spatial_dims)
        
        
        '''fcn 8'''
        # upsample by 2
        fuse1_upsampled = self.upsample_b(fuse1)
        
        # get pool3 pred for net 8
        score_p3 = self.base_net.features[17].val
        
        # crop upsampled score to align with skip connection dimensions
        fuse1_cropped = self._crop(big=fuse1_upsampled, small=score_p3)
        
        # fuse pool3 pred and upsampled fuse1
        fuse2 = fuse1_cropped + score_p3
        
        # bilinear upsample and return
        if self.net=='8': return self.upsample_final(fuse2, out_dim=img_spatial_dims)
        
        
    def _crop(self, big, small):
        '''
        crops big to be the same dimension as small given
        that they are two tensors and the last two dimensions of small
        is smaller than that of big
        
        '''
        
        # get dimensions
        h_small, w_small = small.shape[2:]
    
        return center_crop(img=big, output_size=(h_small, w_small))
    

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
        # upsample layer
        self.val = None
        # pool4 has 512 output channels
        # 30 output channels for each class
        self.conv = nn.Conv2d(in_channels=in_channels, out_channels=n_class, kernel_size=1, stride=1)
        # initialize weights with 0small constant (not 0)
        nn.init.constant_(self.conv.weight, val=0.001)
        # learnable scale 
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        '''
        returns the input as is but stores the result of a convolution.
        
        args
        - x: tensor(batch_size, 512, img_height/16, img_width/16)
        
        '''
        # apply convolution, scale, and store
        self.val = self.scale * self.conv(x)
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
        