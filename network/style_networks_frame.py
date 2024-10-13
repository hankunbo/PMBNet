import torch
import torch.nn as nn 
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import torchvision 
import torchvision.models as models
import torchvision.utils as vutils
import cv2
import kornia
from collections import namedtuple

import math
from torch import nn, einsum

from einops import rearrange
from torchvision.models.vgg import VGG19_Weights

loss_L1 = torch.nn.L1Loss()

###########################################
##   Tools
#------------------------------------------

mean_std = namedtuple("mean_std", ['mean','std'])
vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
vgg_outputs_super = namedtuple("VggOutputs", ['map', 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])





def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    grid = grid.to(x.device)
    vgrid = grid - flo
    
    # scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, )
    return output


def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)


def crop_2d(input, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
    assert input.dim() == 4, 'only support Input(B, C, W, H)'
    B, C, W, H = input.size()
    return input[:, :,
                 crop_left:(W-crop_right),
                 crop_bottom:(H-crop_top)]


class Crop2d(nn.Module):
    def __init__(self, crop_left=0, crop_right=0, crop_top=0, crop_bottom=0):
        super(Crop2d, self).__init__()
        self.crop_left = crop_left
        self.crop_right = crop_right
        self.crop_top = crop_top
        self.crop_bottom = crop_bottom

    def forward(self, input):
        return crop_2d(input,
                       self.crop_left,
                       self.crop_right,
                       self.crop_top,
                       self.crop_bottom)


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat

###########################################
##   Layers and Blocks
#------------------------------------------


class InstanceNorm(nn.Module):
    def __init__(self, epsilon=1e-8):

        """ avoid in-place ops.
            https://discuss.pytorch.org/t/encounter-the-runtimeerror-one-of-the-variables-needed-for-gradient-computation-has-been-modified-by-an-inplace-operation/836/3 """
        
        super(InstanceNorm, self).__init__()
        self.epsilon = epsilon

    def forward(self, x):
        x   = x - torch.mean(x, (2, 3), True)
        tmp = torch.mul(x, x)
        tmp = torch.rsqrt(torch.mean(tmp, (2, 3), True) + self.epsilon)
        return x * tmp



class ResidualBlock(nn.Module):
    def __init__(self, input_channel, output_channel, upsample=True):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(input_channel, output_channel, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(output_channel, output_channel, kernel_size=3, padding=1)
        self.conv_shortcut = nn.Conv2d(input_channel, output_channel, kernel_size=1, bias=False)
        self.relu = nn.LeakyReLU(0.2)
        self.norm = InstanceNorm()
        self.upsample = upsample

    def forward(self, x):
        if self.upsample:
            x = F.interpolate(x, mode='nearest', scale_factor=2)
        x_s = self.conv_shortcut(x)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.norm(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.norm(x)

        return x_s + x

class CBAMLayer(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAMLayer, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            # Conv2d比Linear方便操作
            # nn.Linear(channel, channel // reduction, bias=False)
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            # inplace=True直接替换，节省内存
            nn.ReLU(inplace=True),
            # nn.Linear(channel // reduction, channel,bias=False)
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        # print('max_out:',max_out.shape)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        # print('avg_out:',avg_out.shape)
        a=torch.cat([max_out, avg_out], dim=1)
        # print('a:',a.shape)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        # print('spatial:',spatial_out.shape)
        x = spatial_out * x
        # print('x:',x.shape)
        return x
    
class Enhance_Module(nn.Module):                       ##加入通道注意力机制
    def __init__(self, dim_in, dim_out, rate=1, bn_mom=0.1):
        super(Enhance_Module, self).__init__()
        self.branch1 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 1, 1, padding=0, dilation=rate, bias=True),
            nn.InstanceNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=2 * rate, dilation=2 * rate, bias=True),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=3 * rate, dilation=3 * rate, bias=True),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=4 * rate, dilation=4 * rate, bias=True),
            nn.InstanceNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=5 * rate, dilation=5 * rate, bias=True),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=7 * rate, dilation=7 * rate, bias=True),
            nn.InstanceNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=8 * rate, dilation=8 * rate, bias=True),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=9 * rate, dilation=9 * rate, bias=True),
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=10 * rate, dilation=10 * rate, bias=True),
            nn.InstanceNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch5_conv = nn.Conv2d(dim_in, dim_out, 1, 1, 0, bias=True)
        # self.branch5_in = nn.InstanceNorm2d(dim_out, momentum=bn_mom)
        self.branch5_relu = nn.ReLU(inplace=True)

        self.conv_cat = nn.Sequential(
            nn.Conv2d(dim_out * 5, dim_out, 1, 1, padding=0, bias=True),
            nn.InstanceNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        # print('dim_in:',dim_in)
        # print('dim_out:',dim_out)
        self.cbam2=CBAMLayer(channel=dim_out)
        self.cbam5=CBAMLayer(channel=dim_out*5)

        self.norm = InstanceNorm()

    def cal_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        out = mean_std(feat_mean, feat_std)
        return out

    def AdaIN(self, content_feat, style_feat):
        size = content_feat.size()

        style_mean = style_feat.mean
        style_std = style_feat.std
        self.norm(content_feat)
        normalized_feat = self.norm(content_feat)

        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content_feat, style_feat):
        print("content_feat.size()", content_feat.size(), style_feat.size())
        [b, c, row, col] = content_feat.size()
        [b, c, row_s, col_s] = style_feat.size()
        conv1x1 = self.branch1(content_feat)
        conv3x3_1 = self.branch2(content_feat)
        conv3x3_2 = self.branch3(content_feat)
        conv3x3_3 = self.branch4(content_feat)
        global_feature = torch.mean(content_feat, 2, True)
        global_feature = torch.mean(global_feature, 3, True)
        global_feature = self.branch5_conv(global_feature)
        # global_feature = self.branch5_in(global_feature)
        global_feature = self.branch5_relu(global_feature)
        global_feature = F.interpolate(global_feature, (row, col), None, 'bilinear', True)

        conv1x1_s = self.branch1(style_feat)
        conv3x3_1_s = self.branch2(style_feat)
        conv3x3_2_s = self.branch3(style_feat)  
        conv3x3_3_s = self.branch4(style_feat)
        global_feature_s = torch.mean(style_feat, 2, True)
        global_feature_s = torch.mean(global_feature_s, 3, True)
        global_feature_s = self.branch5_conv(global_feature_s)
        # global_feature_s = self.branch5_in(global_feature_s)
        global_feature_s = self.branch5_relu(global_feature_s)
        global_feature_s = F.interpolate(global_feature_s, (row_s, col_s), None, 'bilinear', True)      

        feature_cat_0 = self.AdaIN(conv1x1, self.cal_mean_std(conv1x1_s))
        feature_cat_1 = self.AdaIN(conv3x3_1, self.cal_mean_std(conv3x3_1_s))
        feature_cat_2 = self.AdaIN(conv3x3_2, self.cal_mean_std(conv3x3_2_s))
        feature_cat_3 = self.AdaIN(conv3x3_3, self.cal_mean_std(conv3x3_3_s))
        feature_cat_global = self.AdaIN(global_feature, self.cal_mean_std(global_feature_s))

        cbam_0 = self.cbam2(feature_cat_0)
        cbam_1 = self.cbam2(feature_cat_1)
        cbam_2 = self.cbam2(feature_cat_2)
        cbam_3 = self.cbam2(feature_cat_3)
        cbam_global = self.cbam2(feature_cat_global)

        feature_cat = torch.cat([cbam_0, cbam_1, cbam_2, cbam_3, cbam_global], dim=1)
        # 加入cbam注意力机制
        cbamaspp=self.cbam5(feature_cat)
        result=self.conv_cat(cbamaspp)
        return result


###########################################
##   Networks
#------------------------------------------


class Vgg19(nn.Module):
    def __init__(self, requires_grad=False):
        super(Vgg19, self).__init__()
        #vgg_pretrained_features = models.vgg19(pretrained=True).features
        vgg_pretrained_features = models.vgg19(weights=VGG19_Weights.DEFAULT).features
        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()
        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    def forward(self, X):
        h = self.slice1(X)
        h_relu1_1 = h
        h = self.slice2(h)
        h_relu2_1 = h
        h = self.slice3(h)
        h_relu3_1 = h
        h = self.slice4(h)
        h_relu4_1 = h
        out = vgg_outputs(h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1)
        return out


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        # VGG
        vgg_pretrained_features = models.vgg19(pretrained=True).features
        self.slice = nn.Sequential()
        for x in range(21):
            self.slice.add_module(str(x), vgg_pretrained_features[x])
            
    def forward(self, cur_frame):
        return self.slice(cur_frame)


class EncoderStyle(nn.Module):
    def __init__(self):
        super(EncoderStyle, self).__init__()
        # VGG
        vgg_pretrained_features = models.vgg19(pretrained=True).features

        self.slice1 = nn.Sequential()
        self.slice2 = nn.Sequential()
        self.slice3 = nn.Sequential()
        self.slice4 = nn.Sequential()

        for x in range(2):
            self.slice1.add_module(str(x), vgg_pretrained_features[x])
        for x in range(2, 7):
            self.slice2.add_module(str(x), vgg_pretrained_features[x])
        for x in range(7, 12):
            self.slice3.add_module(str(x), vgg_pretrained_features[x])
        for x in range(12, 21):
            self.slice4.add_module(str(x), vgg_pretrained_features[x])

    def cal_mean_std(self, feat, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = feat.size()
        N, C = size[:2]
        feat_var = feat.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)

        out = mean_std(feat_mean, feat_std)
        return out

    def forward(self, style):
        h = self.slice1(style)
        h_relu1_1 = self.cal_mean_std(h)

        h = self.slice2(h)
        h_relu2_1 = self.cal_mean_std(h)

        h = self.slice3(h)
        h_relu3_1 = self.cal_mean_std(h)

        h = self.slice4(h)
        h_relu4_1 = self.cal_mean_std(h)

        out = vgg_outputs_super(h, h_relu1_1, h_relu2_1, h_relu3_1, h_relu4_1)
        return out

class FilterPredictor(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(FilterPredictor, self).__init__()
        self.down_sample = nn.Sequential(nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1))
        self.inner_channel = inner_channel
        self.FC = nn.Linear(inner_channel*2,inner_channel*inner_channel)

    def forward(self, content, style):
        content = self.down_sample(content)
        style = self.down_sample(style)

        content = torch.mean(content.view(content.size(0), content.size(1), -1), dim=2)
        style = torch.mean(style.view(style.size(0), style.size(1), -1), dim=2)

        filter = self.FC(torch.cat([content, style],1))
        filter = filter.view(-1, self.inner_channel, self.inner_channel).unsqueeze(3)
        return filter

class KernelFilter(nn.Module):
    def __init__(self, vgg_channel=512, inner_channel=32):
        super(KernelFilter, self).__init__()
        self.down_sample = nn.Sequential(
                        nn.Conv2d(vgg_channel, inner_channel, kernel_size=3, padding=1),
                    )

        self.upsample = nn.Sequential(
                        nn.Conv2d(inner_channel, vgg_channel, kernel_size=3, padding=1),
                    )

        self.F1 = FilterPredictor(vgg_channel, inner_channel)
        self.F2 = FilterPredictor(vgg_channel, inner_channel)

        self.relu = nn.LeakyReLU(0.2)

    def apply_filter(self, input_, filter_):
        ''' input_:  [B, inC, H, W]
            filter_: [B, inC, outC, 1] '''

        B = input_.shape[0]
        input_chunk = torch.chunk(input_, B, dim=0)
        filter_chunt = torch.chunk(filter_, B, dim=0)

        results = []

        for input, filter_ in zip(input_chunk, filter_chunt):
            input = F.conv2d(input, filter_.permute(1,2,0,3), groups=1)
            results.append(input)

        return torch.cat(results,0)

    def forward(self, content, style):
        content_ = self.down_sample(content)

        content_ = self.apply_filter(content_, self.F1(content, style))
        content_ = self.relu(content_)

        content_ = self.apply_filter(content_, self.F2(content, style))

        return content + self.upsample(content_)

class Decoder(nn.Module):
    def __init__(self, dynamic_filter = True,
                 out_channels = 3):
        super(Decoder, self).__init__()

        # self.slice6 = nn.Conv2d(512,512, kernel_size=1, bias=False)
        self.slice5 = nn.Conv2d(512,512, kernel_size=1, bias=False)
        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = nn.Conv2d(64, out_channels, kernel_size=3, padding=1)
        
        self.dynamic_filter = dynamic_filter
        self.norm = InstanceNorm()
        
        init_weights(self)
        
        self.Filter1 = KernelFilter()
        self.Filter2 = KernelFilter()
        self.Filter3 = KernelFilter()

    def AdaIN(self, content_feat, style_feat):
        size = content_feat.size()

        style_mean = style_feat.mean
        style_std = style_feat.std

        normalized_feat = self.norm(content_feat)

        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def AdaIN_filter(self, content_feat, style_feat, style_map):
        size = content_feat.size()

        style_mean = style_feat.mean
        style_std = style_feat.std

        normalized_content = self.norm(content_feat)
        normalized_style = (style_map - style_mean)/style_std

        results = self.Filter1(normalized_content, normalized_style)
        results = self.Filter2(results, normalized_style)
        results = self.Filter3(results, normalized_style)

        return results * style_std.expand(size) + style_mean.expand(size)

    def forward(self, x, style_features=None):

        if self.dynamic_filter:
            h = self.AdaIN_filter(x, style_features.relu4_1, style_features.map)
        else:
            h = self.AdaIN(x, style_features.relu4_1)
        
        h3 = self.slice5(h)
        h4 = x + h3

        h = self.slice4(h4)

        h = self.AdaIN(h, style_features.relu3_1)
        h = self.slice3(h)

        h = self.AdaIN(h, style_features.relu2_1)
        h = self.slice2(h)

        h = self.AdaIN(h, style_features.relu1_1)
        out = self.slice1(h)

        return out


class TransformerNet(nn.Module):
    def __init__(self):
        
        super(TransformerNet, self).__init__()

        # Sub-models
        
        self.Encoder = Encoder()
        self.EncoderStyle = EncoderStyle()
        
        self.Enhance_Module = Enhance_Module(512,512)

        self.DecoderA = Decoder(out_channels = 3)
        self.DecoderM = Decoder(out_channels = 6)
        self.Vgg19 = Vgg19()

        # Parameters
        self.have_delete_vgg = False
        self.num = 0

    def generate_style_features(self, style):
        self.F_style = self.EncoderStyle(style)
        if not self.have_delete_vgg:
            del self.Vgg19
            self.have_delete_vgg = True

    ## ---------------------------------------------------
    ##  Functions for setting the states
    ##  Useful in adversarial training

    def ParamStatic(self):
        for param in self.parameters():
            param.requires_grad = False


    def RGB2Gray(self, image):
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        image = (image*std+mean)

        gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
        gray = gray.expand(image.size())

        gray = (gray-mean)/std

        return gray

    ## ---------------------------------------------------
    ##  Training main function

    def forward(self, input_frame):

        ## Style transfer
        F_content = self.Encoder(self.RGB2Gray(input_frame))
        #F_content = self.Encoder(input_frame)
        F_style = self.F_style

        F_cs = self.Enhance_Module(F_content, F_style.map)

        #print("F_style.shape",F_style.shape)
        print("F_content.shape",F_content.shape)
        print("F_cs.shape",F_cs.shape)

        styled_resultA = self.DecoderA(F_cs, F_style)
        styled_resultM = self.DecoderM(F_cs, F_style)

        return styled_resultA, styled_resultM
