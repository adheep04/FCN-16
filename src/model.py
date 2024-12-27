from torch import nn
import torch
from torchvision.models import vgg16, VGG16_Weights

class FCN(nn.Module):
    '''
    class for a fully convolutional network described in the paper "Fully Convolutional Networks for Semantic Segmentation"
    uses part of a pretrained base model and includes upsampling using deconvolution
    
    args:
    - n_class: int
        - number of classes to identify
    - net: str in ['32', '16', '8']
        - which fcn variant this is
    '''
    def __init__(self, n_class, net='32'):        
        super().__init__()
        self.net = net
        # pretrained feature extractor
        # replace last 3 fully connected with convolution layers
        self.base_net = self._adapt_base_net(vgg16(weights=VGG16_Weights.DEFAULT), n_class)
        
        # initialize post-pool skip connection 
        pool3_conv = SkipConnection(in_channels=256) # result is (b, c, h//8, w//8)
        pool4_conv = SkipConnection(in_channels=512) # result is (b, c, h//16, w//16)
        
        # insert skip connection conv1x1s in sequential after pool3 and pool4
        self.base_net.features.insert(17, pool3_conv)
        self.base_net.features.insert(25, pool4_conv)
        
        # upsamples pool5 pred to fuse with pool4 pred in fcn16
        self.upsample2_1 = self._bilinear_weight_init(nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4,stride=2, bias=False))
        # upsamples the fuse result of pool4 and pool5 pred to fuse with pool3 pred in fcn8
        self.upsample2_2 = self._bilinear_weight_init(nn.ConvTranspose2d(in_channels=n_class, out_channels=n_class, kernel_size=4,stride=2, bias=False))
        # final upsample is essentially a bilinear interpolation and can be used by either net given the original spatial dim of image
        self.upsample_final = FixedInterpolation()
        
        
    
    def forward(self, x):
        '''
        forward pass in end-to-end image segmentation model FCN
        
        args:
        - x: tensor(batch_size, channel_size, height, width)
        
        output:
        - tensor(batch_size, class_size, height, width)
        
        '''
        
        
        '''fcn 32'''
        # get img spatial dimensions
        img_spatial_dims = (x.shape[-2], x.shape[-1])
        
        # pass through base vgg16 to get coarse class predictions
        pool5_pred = self.base_net(x)
        
        # if net is fcn32, upsample back to original spatial dimensions and return
        if self.net=='32': return self.upsample_final(pool5_pred, out_dim=img_spatial_dims)
        
        
        '''fcn 16'''
        # upsample by 2
        pool5_up = self.upsample2_1(pool5_pred)
        
        # get stored pool4 pred for net 16 and 8
        pool4_pred = self.base_net.features[17].val
        
        # validate both dimensions are the same for fuse (since it uses element-wise addition)
        assert pool4_pred.shape == pool5_up.shape, "pool4 pred and upsampled pool5 shape mismatch"
        
        # fuse both (sum them)
        fuse_1 = self._fuse([pool5_up, pool4_pred])
        
        # bilinear upsample back to image spatial dim and return
        if self.net=='16': return self.upsample_final(fuse_1, out_dim=img_spatial_dims)
        
        
        '''fcn 8'''
        # upsample by 2
        fuse_1_up = self.upsample2_2(fuse_1)
        
        # get pool3 pred for net 8
        pool3_pred = self.base_net.features[25].val
        
        # validate
        assert pool3_pred.shape == fuse_1_up.shape, "pool3 pred and upsampled fuse_1 shape mismatch"
        
        # fuse pool3 pred and upsampled fuse_1
        fuse_2 = self._fuse([fuse_1_up, pool3_pred])
        
        # bilinear upsample and return
        if self.net=='8': return self.upsample_final(fuse_2, out_dim=img_spatial_dims)
        
    
        
    def _bilinear_weight_init(conv):
        pass
    
    def _adapt_base_net(self, model, n_class):
        '''
        replaces the last 3 linear layers with convolution layers
        maintaining trained parameter values. 
        
        adds 1x1 conv layer after pool4 for future image reconstruction
        '''

        # get weights and biases from last 2 fully connected layers in VGG-16
        linear1_weights = model.classifier[0].weight.data
        linear2_weights = model.classifier[3].weight.data
        linear1_bias = model.classifier[0].bias.data
        linear2_bias = model.classifier[3].bias.data

        # reshape to fit convolution layer dimensions
        reshaped_weights1 = linear1_weights.reshape(4096, 512, 7, 7)
        reshaped_weights2 = linear2_weights.reshape(4096, 4096, 1, 1)
        zero_weights = torch.zeros(n_class, 4096, 1, 1)

        # use pytorch's convolution layer class to initialize conv layers
        conv1 = nn.Conv2d(in_channels=512, out_channels=4096, kernel_size=7)
        conv2 = nn.Conv2d(in_channels=4096, out_channels=4096, kernel_size=1)
        conv3 = nn.Conv2d(in_channels=4096, out_channels=n_class, kernel_size=1)
        
        # replace default weights with trained reshaped weights
        conv1.weight.data = reshaped_weights1
        conv2.weight.data = reshaped_weights2
        
        # class prediction weights initialized to 0
        conv3.weight.data = zero_weights
        
        # replace biases
        conv1.bias.data = linear1_bias
        conv2.bias.data = linear2_bias

        # replace linear layers with convlayers
        model.classifier[0] = conv1
        model.classifier[3] = conv2
        model.classifier[6] = conv3
        
        # remove average pooling layer
        model.avgpool = nn.Identity()
    
    def _fuse(self, *preds):
        '''
        given prediction tensors from different layers in the net, sums them element wise
        
        '''
        # initialize output to first tensor (pool5 pred)
        sum = preds[0]
        # add the remaining
        for i in range(1, len(preds)):
            sum += preds[i]
        return sum

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