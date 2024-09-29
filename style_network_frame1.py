import torch
import torch.nn as nn 
from torch.nn import init
from torch.autograd import Variable
import torch.nn.functional as F
import torch.optim as optim

import torchvision 
import torchvision.models as models
import torchvision.utils as vutils

import kornia
from collections import namedtuple
import math
from torch import nn, einsum

from einops import rearrange



###########################################
##   Tools
#------------------------------------------

mean_std = namedtuple("mean_std", ['mean','std'])
vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
vgg_outputs_super = namedtuple("VggOutputs", ['map', 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
# SSIM = SSIM()





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

###########################################
##   Transformer Transform
#------------------------------------------
# def pair(x):
#     return (x, x) if not isinstance(x, tuple) else x

# def expand_dim(t, dim, k):
#     t = t.unsqueeze(dim = dim)
#     expand_shape = [-1] * len(t.shape)
#     expand_shape[dim] = k
#     return t.expand(*expand_shape)

# def rel_to_abs(x):
#     b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
#     dd = {'device': device, 'dtype': dtype}
#     col_pad = torch.zeros((b, h, l, 1), **dd)
#     x = torch.cat((x, col_pad), dim = 3)
#     flat_x = rearrange(x, 'b h l c -> b h (l c)')
#     flat_pad = torch.zeros((b, h, l - 1), **dd)
#     flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
#     final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
#     final_x = final_x[:, :, :l, (l-1):]
#     return final_x

# def relative_logits_1d(q, rel_k):
#     b, heads, h, w, dim = q.shape
#     logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
#     logits = rearrange(logits, 'b h x y r -> b (h x) y r')
#     logits = rel_to_abs(logits)
#     logits = logits.reshape(b, heads, h, w, w)
#     logits = expand_dim(logits, dim = 3, k = h)
#     return logits

# # positional embeddings

# class AbsPosEmb(nn.Module):
#     def __init__(
#         self,
#         fmap_size,
#         dim_head,

#     ):
#         super().__init__()
#         height, width = pair(fmap_size)
#         scale = dim_head ** -0.5
#         self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
#         self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

#     def forward(self, q):
#         # b, c, h, w = k.shape
#         # scale = 128 ** -0.5
#         # self.height = nn.Parameter(torch.randn(h, 128) * scale)
#         # self.width = nn.Parameter(torch.randn(w, 128) * scale)

#         emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
#         emb = rearrange(emb, ' h w d -> (h w) d')
#         logits = einsum('b h i d, j d -> b h i j', q, emb)
#         return logits


# class RelPosEmb(nn.Module):
#     def __init__(
#         self,
#         fmap_size,
#         dim_head
#     ):
#         super().__init__()
#         height, width = pair(fmap_size)
#         scale = dim_head ** -0.5
#         self.fmap_size = fmap_size
#         self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
#         self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

#     def forward(self, q):
#         h, w = self.fmap_size

#         q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
#         rel_logits_w = relative_logits_1d(q, self.rel_width)
#         rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

#         q = rearrange(q, 'b h x y d -> b h y x d')
#         rel_logits_h = relative_logits_1d(q, self.rel_height)
#         rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
#         return rel_logits_w + rel_logits_h

# # classes

# class MSANet(nn.Module):
#     def __init__(
#         self,
#         *,
#         dim = 512,
#         content_feat_size = 32,
#         heads = 4,
#         dim_head = 128,
#         rel_pos_emb = False
#     ):
#         super().__init__()
#         self.heads = heads
#         self.scale = dim_head ** -0.5
#         inner_dim = heads * dim_head

#         self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
#         self.to_k = nn.Conv2d(dim, inner_dim, 1, bias = False)
#         self.to_v = nn.Conv2d(dim, inner_dim, 1, bias = False)

#         rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
#         self.pos_emb = rel_pos_class(content_feat_size, dim_head)

#     def forward(self, content_feat, style_feat):
#         heads, b, c, h, w = self.heads, *content_feat.shape
#         _, _, h_s, w_s = style_feat.shape
        
#         if h_s !=32 and w_s!=32:
#             style_feat = F.interpolate(style_feat, size=[32,32], mode="bilinear")


        

#         q = self.to_q(mean_variance_norm(content_feat))
#         k = self.to_k(mean_variance_norm(style_feat))
#         v = self.to_v(style_feat)
        
#         # q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))
#         q = rearrange(q, 'b (h d) x y -> b h (x y) d', h = heads)
#         k = rearrange(k, 'b (h d) x y -> b h (x y) d', h = heads)
#         v = rearrange(v, 'b (h d) x y -> b h (x y) d', h = heads)
#         # print(q.shape,k.shape,v.shape)

#         # q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))

#         q *= self.scale

#         sim = einsum('b h i d, b h j d -> b h i j', q, k)
#         sim += self.pos_emb(q)

#         attn = sim.softmax(dim = -1)

#         out = einsum('b h i j, b h j d -> b h i d', attn, v)
#         out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
#         return out


###########################################
##   Enhance module
#------------------------------------------

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
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=6 * rate, dilation=6 * rate, bias=True),
            nn.InstanceNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch3 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=12 * rate, dilation=12 * rate, bias=True),
            nn.InstanceNorm2d(dim_out, momentum=bn_mom),
            nn.ReLU(inplace=True),
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(dim_in, dim_out, 3, 1, padding=18 * rate, dilation=18 * rate, bias=True),
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

        normalized_feat = self.norm(content_feat)

        return normalized_feat * style_std.expand(size) + style_mean.expand(size)

    def forward(self, content_feat, style_feat):
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

# class SANet(nn.Module):
#     def __init__(self, input_channel):
#         super(SANet, self).__init__()
#         self.f = nn.Conv2d(input_channel, input_channel, (1, 1))
#         self.g = nn.Conv2d(input_channel, input_channel, (1, 1))
#         self.h = nn.Conv2d(input_channel, input_channel, (1, 1))
#         self.sm = nn.Softmax(dim = -1)
#         self.out_conv = nn.Conv2d(input_channel, input_channel, (1, 1))
#     def forward(self, content_feat, style_feat):
#         F = self.f(mean_variance_norm(content_feat))
#         G = self.g(mean_variance_norm(style_feat))
#         H = self.h(style_feat)
#         b, c, h, w = F.size()
#         F = F.view(b, -1, w * h).permute(0, 2, 1)
#         b, c, h, w = G.size()
#         G = G.view(b, -1, w * h)
#         S = torch.bmm(F, G)
#         S = self.sm(S)
#         b, c, h, w = H.size()
#         H = H.view(b, -1, w * h)
#         out = torch.bmm(H, S.permute(0, 2, 1))
#         b, c, h, w = content_feat.size()
#         out = out.view(b, c, h, w)
#         out = self.out_conv(out)
#         out += content_feat
#         return out

class Decoder(nn.Module):
    def __init__(self, style_attention=True,
        *,
        dim = 512,
        content_feat_size = 32,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False,
    ):
        super(Decoder, self).__init__()

        # self.slice6 = nn.Conv2d(512,512, kernel_size=1, bias=False)
        self.slice5 = nn.Conv2d(512,512, kernel_size=1, bias=False)
        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.norm = InstanceNorm()


        self.style_attention = style_attention
        self.Enhance_Module = Enhance_Module(512,512)



        init_weights(self)

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
        # print(x.shape)

        # if self.style_attention: 
            # h =  self.slice6(x)                                           #   style transfer transformer layer
            # h = self.MSANet4_1(x, style_features.map)
            # h = self.AdaIN(h, style_features.relu4_1)
            # h3 = self.slice5(h2)
            # h = h3 + h
        h1 = self.Enhance_Module(x, style_features.map)

        h2 = self.AdaIN(h1, style_features.relu4_1)
        h3 = self.slice5(h2)
        h4 = x + h3
        
        h = self.slice4(h4)

        h = self.AdaIN(h, style_features.relu3_1)
        h = self.slice3(h)

        h = self.AdaIN(h, style_features.relu2_1)
        h = self.slice2(h)

        h = self.AdaIN(h, style_features.relu1_1)
        h = self.slice1(h)

        return h



class TransformerNet(nn.Module):
    def __init__(self, style_attention=True, train_only_decoder=False,
                       style_content_loss=True, spa_loss=True, exp_loss=True, ssim_loss=True, recon_loss=True, recon_ssim_loss=True, relax_style=True):
        
        super(TransformerNet, self).__init__()

        # Sub-models
        self.Decoder = Decoder(style_attention=style_attention)
        self.Encoder = Encoder()
        self.EncoderStyle = EncoderStyle()
        self.Vgg19 = Vgg19()

        self.have_delete_vgg = False

    def generate_style_features(self, style):
        self.F_style = self.EncoderStyle(style)
        if not self.have_delete_vgg:
            del self.Vgg19
            self.have_delete_vgg = True

    def RGB2Gray(self, image):
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        image = (image*std+mean)

        gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
        gray = gray.expand(image.size())

        gray = (gray-mean)/std

        return gray

    def forward(self, input_frame):
        F_content = self.Encoder(self.RGB2Gray(input_frame))
        return self.Decoder(F_content, self.F_style)

