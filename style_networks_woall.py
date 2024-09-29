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
from ssim_loss import SSIM

import math
from torch import nn, einsum

from einops import rearrange




###########################################
##   Tools
#------------------------------------------

mean_std = namedtuple("mean_std", ['mean','std'])
vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
vgg_outputs_super = namedtuple("VggOutputs", ['map', 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
SSIM = SSIM()





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
def pair(x):
    return (x, x) if not isinstance(x, tuple) else x

def expand_dim(t, dim, k):
    t = t.unsqueeze(dim = dim)
    expand_shape = [-1] * len(t.shape)
    expand_shape[dim] = k
    return t.expand(*expand_shape)

def rel_to_abs(x):
    b, h, l, _, device, dtype = *x.shape, x.device, x.dtype
    dd = {'device': device, 'dtype': dtype}
    col_pad = torch.zeros((b, h, l, 1), **dd)
    x = torch.cat((x, col_pad), dim = 3)
    flat_x = rearrange(x, 'b h l c -> b h (l c)')
    flat_pad = torch.zeros((b, h, l - 1), **dd)
    flat_x_padded = torch.cat((flat_x, flat_pad), dim = 2)
    final_x = flat_x_padded.reshape(b, h, l + 1, 2 * l - 1)
    final_x = final_x[:, :, :l, (l-1):]
    return final_x

def relative_logits_1d(q, rel_k):
    b, heads, h, w, dim = q.shape
    logits = einsum('b h x y d, r d -> b h x y r', q, rel_k)
    logits = rearrange(logits, 'b h x y r -> b (h x) y r')
    logits = rel_to_abs(logits)
    logits = logits.reshape(b, heads, h, w, w)
    logits = expand_dim(logits, dim = 3, k = h)
    return logits

# positional embeddings

class AbsPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head,

    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.height = nn.Parameter(torch.randn(height, dim_head) * scale)
        self.width = nn.Parameter(torch.randn(width, dim_head) * scale)

    def forward(self, q):
        # b, c, h, w = k.shape
        # scale = 128 ** -0.5
        # self.height = nn.Parameter(torch.randn(h, 128) * scale)
        # self.width = nn.Parameter(torch.randn(w, 128) * scale)

        emb = rearrange(self.height, 'h d -> h () d') + rearrange(self.width, 'w d -> () w d')
        emb = rearrange(emb, ' h w d -> (h w) d')
        logits = einsum('b h i d, j d -> b h i j', q, emb)
        return logits

class RelPosEmb(nn.Module):
    def __init__(
        self,
        fmap_size,
        dim_head
    ):
        super().__init__()
        height, width = pair(fmap_size)
        scale = dim_head ** -0.5
        self.fmap_size = fmap_size
        self.rel_height = nn.Parameter(torch.randn(height * 2 - 1, dim_head) * scale)
        self.rel_width = nn.Parameter(torch.randn(width * 2 - 1, dim_head) * scale)

    def forward(self, q):
        h, w = self.fmap_size

        q = rearrange(q, 'b h (x y) d -> b h x y d', x = h, y = w)
        rel_logits_w = relative_logits_1d(q, self.rel_width)
        rel_logits_w = rearrange(rel_logits_w, 'b h x i y j-> b h (x y) (i j)')

        q = rearrange(q, 'b h x y d -> b h y x d')
        rel_logits_h = relative_logits_1d(q, self.rel_height)
        rel_logits_h = rearrange(rel_logits_h, 'b h x i y j -> b h (y x) (j i)')
        return rel_logits_w + rel_logits_h

# classes

class MSANet(nn.Module):
    def __init__(
        self,
        *,
        dim = 512,
        content_feat_size = 32,
        heads = 4,
        dim_head = 128,
        rel_pos_emb = False
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias = False)
        self.to_v = nn.Conv2d(dim, inner_dim, 1, bias = False)

        rel_pos_class = AbsPosEmb if not rel_pos_emb else RelPosEmb
        self.pos_emb = rel_pos_class(content_feat_size, dim_head)

    def forward(self, content_feat, style_feat):
        heads, b, c, h, w = self.heads, *content_feat.shape
        _, _, h_s, w_s = style_feat.shape
        
        if h_s !=32 and w_s!=32:
            style_feat = F.interpolate(style_feat, size=[32,32], mode="bilinear")


        

        q = self.to_q(mean_variance_norm(content_feat))
        k = self.to_k(mean_variance_norm(style_feat))
        v = self.to_v(style_feat)
        
        # q, k, v = map(lambda t: rearrange(t, 'b (h d) x y -> b h (x y) d', h = heads), (q, k, v))
        q = rearrange(q, 'b (h d) x y -> b h (x y) d', h = heads)
        k = rearrange(k, 'b (h d) x y -> b h (x y) d', h = heads)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h = heads)

        q *= self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        sim += self.pos_emb(q)

        attn = sim.softmax(dim = -1)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x = h, y = w)
        return out

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

class SANet(nn.Module):
    def __init__(self, input_channel):
        super(SANet, self).__init__()
        self.f = nn.Conv2d(input_channel, input_channel, (1, 1))
        self.g = nn.Conv2d(input_channel, input_channel, (1, 1))
        self.h = nn.Conv2d(input_channel, input_channel, (1, 1))
        self.sm = nn.Softmax(dim = -1)
        self.out_conv = nn.Conv2d(input_channel, input_channel, (1, 1))
    def forward(self, content_feat, style_feat):
        F = self.f(mean_variance_norm(content_feat))
        G = self.g(mean_variance_norm(style_feat))
        H = self.h(style_feat)
        b, c, h, w = F.size()
        F = F.view(b, -1, w * h).permute(0, 2, 1)
        b, c, h, w = G.size()
        G = G.view(b, -1, w * h)
        S = torch.bmm(F, G)
        S = self.sm(S)
        b, c, h, w = H.size()
        H = H.view(b, -1, w * h)
        out = torch.bmm(H, S.permute(0, 2, 1))
        b, c, h, w = content_feat.size()
        out = out.view(b, c, h, w)
        out = self.out_conv(out)
        out += content_feat
        return out

class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        # self.slice6 = nn.Conv2d(512,512, kernel_size=1, bias=False)
        # self.slice5 = nn.Conv2d(512,512, kernel_size=1, bias=False)
        self.slice4 = ResidualBlock(512, 256)
        self.slice3 = ResidualBlock(256, 128)
        self.slice2 = ResidualBlock(128, 64)
        self.slice1 = nn.Conv2d(64, 3, kernel_size=3, padding=1)

        self.norm = InstanceNorm()


        # self.style_attention = style_attention
        # if style_attention:
        #     self.MSANet4_1 = MSANet(dim = dim,
        #                         content_feat_size = content_feat_size,
        #                         heads = heads,
        #                         dim_head = dim_head,
        #                         rel_pos_emb = rel_pos_emb)
        #     self.SANet4_1 = SANet(512)


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

        # if self.style_attention: #   style transfer transformer layer
        #     h1 = self.MSANet4_1(x, style_features.map)
        #     h2 = self.SANet4_1(x, style_features.map)
        #     h3 = h1 + h2
        #     h4 = self.AdaIN(h3, style_features.relu4_1)
        #     h5 = self.slice5(h4)
        #     h = x + h5

        h = self.AdaIN(x, style_features.relu4_1)
        h = self.slice4(h)

        h = self.AdaIN(h, style_features.relu3_1)
        h = self.slice3(h)

        h = self.AdaIN(h, style_features.relu2_1)
        h = self.slice2(h)

        h = self.AdaIN(h, style_features.relu1_1)
        h = self.slice1(h)

        return h


class TransformerNet(nn.Module):
    def __init__(self, train_only_decoder=False,
                       style_content_loss=True, spa_loss=True, exp_loss=True, ssim_loss=True, recon_loss=True, recon_ssim_loss=True, relax_style=True):
        
        super(TransformerNet, self).__init__()



        # Sub-models
        self.Decoder = Decoder()
        self.Encoder = Encoder()
        self.EncoderStyle = EncoderStyle()
        self.Vgg19 = Vgg19()

        if train_only_decoder:
            for param in self.Encoder.parameters():
                param.requires_grad = False

            for param in self.EncoderStyle.parameters():
                param.requires_grad = False

        # Other functions and tools
        self.MSE = nn.MSELoss()
        self.Padding = nn.ReflectionPad2d((32,32,32,32))
        self.Cropping = Crop2d(32,32,32,32)
        self.gauss = kornia.filters.GaussianBlur2d((101, 101), (50.5, 50.5))

        # Parameters
        self.flow_scale = 8
        self.flow_iter = 16
        self.flow_max = 20
        self.flow_lr = 16

        self.use_style_loss = style_content_loss
        self.use_content_loss = style_content_loss
        self.use_recon_loss = recon_loss
        self.use_recon_ssim_loss = recon_ssim_loss
        self.relax_style = relax_style

    ## ---------------------------------------------------
    ##  Functions for setting the states
    ##  Useful in adversarial training

    def ParamStatic(self):
        for param in self.parameters():
            param.requires_grad = False

    def ParamActive(self):
        for param in self.Encoder.parameters():
            param.requires_grad = True

        for param in self.Decoder.parameters():
            param.requires_grad = True

        for param in self.EncoderStyle.parameters():
            param.requires_grad = True

    ## ---------------------------------------------------
    ##  Style loss, Content loss, Desaturation

    def style_loss(self, features_coded_Image, features_style):
        style_loss = 0.
        for ft_x, ft_s in zip(features_coded_Image, features_style):
            mean_x, var_x = calc_mean_std(ft_x)
            mean_style, var_style = calc_mean_std(ft_s)

            style_loss = style_loss + self.MSE(mean_x, mean_style)
            style_loss = style_loss + self.MSE(var_x, var_style)

        return style_loss

    def content_loss(self, features_coded_Image, features_content):
        content_loss = self.MSE(features_coded_Image.relu4_1, features_content.relu4_1)
        return content_loss

    def RGB2Gray(self, image):
        mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

        image = (image*std+mean)

        gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
        gray = gray.expand(image.size())

        gray = (gray-mean)/std

        return gray

    ## ---------------------------------------------------
    ##  Debug tool for saving results

    def save_figure(self, img, name):
        img = img.data.clone()
        mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
        img = (img*std+mean)
        vutils.save_image(img,'result.png')

    ## ---------------------------------------------------
    ##  Functions for the proposed Relaxed Style Loss

    def get_input_optimizer(self, input_img):
        return optim.SGD([input_img.requires_grad_()], lr=self.flow_lr, momentum=0.9)

    def smooth_flow(self, flow, H, W):
        flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners = True)
        flow = torch.tanh(flow) * self.flow_max
        flow = self.gauss(flow)
        return flow

    ## ---------------------------------------------------
    ##  Inference main function

    def validation(self, cur_frame, style):
        F_cur = self.Encoder(cur_frame)
        F_style = self.EncoderStyle(style)
        return self.Decoder(F_cur, F_style)

    ## ---------------------------------------------------
    ##  Training main function

    def forward(self, content, style):

        ## Content image desaturation
        gray_content = self.RGB2Gray(content)

        ## Style transfer
        F_content = self.Encoder(content)
        F_style = self.EncoderStyle(style)
        styled_result = self.Decoder(F_content, F_style)

        ## Style loss and content loss

        # Get ground truth style/content features
        if self.use_content_loss or self.use_style_loss:
            F_styled = self.Vgg19(styled_result)

            if self.use_content_loss:
                F_content_gt = self.Vgg19(gray_content)

            if self.use_style_loss:
                F_style_gt = self.Vgg19(style)

        # Content loss
        if self.use_content_loss:
            content_loss = self.content_loss(F_styled, F_content_gt)
        else:
            content_loss = 0.

        # Style loss
        if self.use_style_loss:
            if self.relax_style:
                ori_style_loss = self.style_loss(F_styled, F_style_gt)
                
                ''' The proposed Relaxed Style Loss '''

                # Init flow
                B,C,H,W = style.shape
                Flow = torch.zeros([B,2,H//self.flow_scale,W//self.flow_scale]).to(style.device)

                # Optimizer
                optimizer = self.get_input_optimizer(Flow)

                # Records
                best_Bounded_Flow = None
                min_style_loss = ori_style_loss.item()
                min_iter = -1

                # Target loss
                static_F_style = vgg_outputs(F_styled.relu1_1.detach(), 
                                             F_styled.relu2_1.detach(), 
                                             F_styled.relu3_1.detach(), 
                                             F_styled.relu4_1.detach())
                
                tmp_style = style.detach()

                ''' We need to find the best <Flow> to minimize <style_loss>.
                    First, <Flow> is gaussian-smoothed by <self.smooth_flow>.
                    Then, the style image is warped by the flow.
                    Finally, we calculate the <style_loss> and do back-propagation. '''

                for i in range(self.flow_iter):
                    optimizer.zero_grad()

                    # Gaussian-smooth the flow
                    Bounded_Flow = self.smooth_flow(Flow, H, W)

                    # Warp the style image using the flow
                    warpped_tmp_style = warp(tmp_style, Bounded_Flow)

                    # Calculate style loss
                    tmp_F_style_gt = self.Vgg19(warpped_tmp_style)
                    style_loss = self.style_loss(static_F_style, tmp_F_style_gt)
                    
                    style_loss.backward()
                    optimizer.step()

                    if style_loss < min_style_loss:
                        min_style_loss = style_loss.item()
                        best_Bounded_Flow = Bounded_Flow.detach()
                        min_iter = i

                if min_iter != -1:
                    robust_tmp_style = warp(tmp_style, best_Bounded_Flow)
                    robust_F_style_gt = self.Vgg19(robust_tmp_style)
                    new_style_loss = self.style_loss(F_styled, robust_F_style_gt)

                    del best_Bounded_Flow

                else:
                    robust_tmp_style = style
                    new_style_loss = ori_style_loss
            else:
                new_style_loss = self.style_loss(F_styled, F_style_gt)
                ori_style_loss = 0.
                robust_tmp_style = None
        else:
            ori_style_loss = 0.
            new_style_loss = 0.
            robust_tmp_style = None

        ## Reconstruction loss
        if self.use_recon_loss:
            recon_content = self.Decoder(F_content, self.EncoderStyle(content))
            recon_style = self.Decoder(self.Encoder(self.RGB2Gray(style)), F_style)
            recon_loss = torch.mean(torch.abs(recon_content-content)) + torch.mean(torch.abs(recon_style-style))
            recon_ssim_loss = 2-SSIM(recon_content,content)-SSIM(recon_style,style)            

            
        else:
            recon_loss = 0.
            recon_content = None
            recon_style = None
        




        return styled_result, robust_tmp_style, recon_content, recon_style, \
                    content_loss, new_style_loss, recon_loss, recon_ssim_loss, ori_style_loss



   