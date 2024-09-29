import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from collections import namedtuple
from loss.ssim_loss import SSIM
import torchvision 
import torchvision.models as models
import kornia
from torchvision.models.vgg import VGG19_Weights

from loss.vgg import VGG19
VGG_19 = VGG19(requires_grad=False).to("cuda")

mean_std = namedtuple("mean_std", ['mean','std'])
vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
vgg_outputs_super = namedtuple("VggOutputs", ['map', 'relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])
SSIM = SSIM()

loss_L1 = torch.nn.L1Loss()
MSE = torch.nn.MSELoss()
gauss = kornia.filters.GaussianBlur2d((101, 101), (50.5, 50.5))

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

def crt_loss(prediction, net_gt, net_in):
    prediction_main = prediction[:,:3,:,:]
    prediction_minor = prediction[:,3:,:,:]
    
    diff_map_main,_ = torch.max(torch.abs(prediction_main - net_gt) / (net_in+1e-1), dim=1, keepdim=True)
    diff_map_minor,_ = torch.max(torch.abs(prediction_minor - net_gt) / (net_in+1e-1), dim=1, keepdim=True)
    #diff_map_main,_ = torch.max(torch.abs(prediction_main - net_gt), dim=1, keepdim=True)
    #diff_map_minor,_ = torch.max(torch.abs(prediction_minor - net_gt), dim=1, keepdim=True)

    confidence_map = torch.lt(diff_map_main, diff_map_minor).repeat(1,3,1,1).float()
    crt_loss = loss_L1(prediction_main*confidence_map, net_gt*confidence_map) \
                        + loss_L1(prediction_minor*(1-confidence_map), net_gt*(1-confidence_map))
    
    return crt_loss

def content_loss(features_coded_Image, features_content):
    content_loss = MSE(features_coded_Image.relu4_1, features_content.relu4_1)
    return content_loss

def get_input_optimizer(input_img):
    return optim.SGD([input_img.requires_grad_()], lr=16, momentum=0.9)

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
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, align_corners=True)
    return output

def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std

def style_loss(features_coded_Image, features_style):
    style_loss = 0.
    for ft_x, ft_s in zip(features_coded_Image, features_style):
        mean_x, var_x = calc_mean_std(ft_x)
        mean_style, var_style = calc_mean_std(ft_s)

        style_loss = style_loss + MSE(mean_x, mean_style)
        style_loss = style_loss + MSE(var_x, var_style)

    return style_loss

def smooth_flow(flow, H, W):
    flow_max = 20
    flow = F.interpolate(flow, (H, W), mode='bilinear', align_corners = True)
    flow = torch.tanh(flow) * flow_max
    flow = gauss(flow)
    return flow

def style_total_loss(style, F_styled, F_style_gt, relax_style=False):
    flow_scale = 8
    flow_iter = 16
    
    if relax_style:
        ori_style_loss = style_loss(F_styled, F_style_gt)

        # Init flow
        B,C,H,W = style.shape
        Flow = torch.zeros([B,2,H//flow_scale,W//flow_scale]).to(style.device)

        # Optimizer
        optimizer = get_input_optimizer(Flow)

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

        for i in range(flow_iter):
            optimizer.zero_grad()

            # Gaussian-smooth the flow
            Bounded_Flow = smooth_flow(Flow, H, W)

            # Warp the style image using the flow
            warpped_tmp_style = warp(tmp_style, Bounded_Flow)

            # Calculate style loss
            tmp_F_style_gt = Vgg19(warpped_tmp_style)
            styleloss = style_loss(static_F_style, tmp_F_style_gt)
            
            styleloss.backward()
            optimizer.step()

            if styleloss < min_style_loss:
                min_style_loss = styleloss.item()
                best_Bounded_Flow = Bounded_Flow.detach()
                min_iter = i

        if min_iter != -1:
            robust_tmp_style = warp(tmp_style, best_Bounded_Flow)
            robust_F_style_gt = Vgg19(robust_tmp_style)
            new_style_loss = styleloss(F_styled, robust_F_style_gt)

            del best_Bounded_Flow

        else:
            robust_tmp_style = style
            new_style_loss = ori_style_loss
    else:
        new_style_loss = style_loss(F_styled, F_style_gt)
        ori_style_loss = 0.
        robust_tmp_style = None

    return robust_tmp_style, new_style_loss, ori_style_loss

def normalize_batch(batch):
    # Normalize batch using ImageNet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(1, -1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(1, -1, 1, 1)
    return (batch - mean) / std

# define loss function 
def compute_error(real,fake):
    # return tf.reduce_mean(tf.abs(fake-real))
    return torch.mean(torch.abs(fake-real))

def Lp_loss(x, y):
    vgg_real = VGG_19(normalize_batch(x))
    vgg_fake = VGG_19(normalize_batch(y))
    p0 = compute_error(normalize_batch(x), normalize_batch(y))
    
    content_loss_list = []
    content_loss_list.append(p0)
    feat_layers = {'conv1_2' : 1.0/2.6, 'conv2_2' : 1.0/4.8, 'conv3_2': 1.0/3.7, 'conv4_2':1.0/5.6, 'conv5_2':10.0/1.5}

    for layer, w in feat_layers.items():
        pi = compute_error(vgg_real[layer], vgg_fake[layer])
        content_loss_list.append(w * pi)
    
    content_loss = torch.sum(torch.stack(content_loss_list))

    return content_loss