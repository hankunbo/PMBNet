import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.utils as vutils
from torch.autograd import Variable

import cv2
import numpy as np

import random
from collections import namedtuple



'''Zero-DCE Spatial Consistency Loss'''
class L_spa(nn.Module):

    def __init__(self):
        super(L_spa, self).__init__()
        # print(1)kernel = torch.FloatTensor(kernel).unsqueeze(0).unsqueeze(0)
        kernel_left = torch.FloatTensor( [[0,0,0],[-1,1,0],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_right = torch.FloatTensor( [[0,0,0],[0,1,-1],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_up = torch.FloatTensor( [[0,-1,0],[0,1, 0 ],[0,0,0]]).cuda().unsqueeze(0).unsqueeze(0)
        kernel_down = torch.FloatTensor( [[0,0,0],[0,1, 0],[0,-1,0]]).cuda().unsqueeze(0).unsqueeze(0)
        self.weight_left = nn.Parameter(data=kernel_left, requires_grad=False)
        self.weight_right = nn.Parameter(data=kernel_right, requires_grad=False)
        self.weight_up = nn.Parameter(data=kernel_up, requires_grad=False)
        self.weight_down = nn.Parameter(data=kernel_down, requires_grad=False)
        self.pool = nn.AvgPool2d(4)
    def forward(self, org , enhance ):
        b,c,h,w = org.shape

        org_mean = torch.mean(org,1,keepdim=True)
        enhance_mean = torch.mean(enhance,1,keepdim=True)

        org_pool =  self.pool(org_mean)			
        enhance_pool = self.pool(enhance_mean)	

        weight_diff =torch.max(torch.FloatTensor([1]).cuda() + 10000*torch.min(org_pool - torch.FloatTensor([0.3]).cuda(),torch.FloatTensor([0]).cuda()),torch.FloatTensor([0.5]).cuda())
        E_1 = torch.mul(torch.sign(enhance_pool - torch.FloatTensor([0.5]).cuda()) ,enhance_pool-org_pool)


        D_org_letf = F.conv2d(org_pool , self.weight_left, padding=1)
        D_org_right = F.conv2d(org_pool , self.weight_right, padding=1)
        D_org_up = F.conv2d(org_pool , self.weight_up, padding=1)
        D_org_down = F.conv2d(org_pool , self.weight_down, padding=1)

        D_enhance_letf = F.conv2d(enhance_pool , self.weight_left, padding=1)
        D_enhance_right = F.conv2d(enhance_pool , self.weight_right, padding=1)
        D_enhance_up = F.conv2d(enhance_pool , self.weight_up, padding=1)
        D_enhance_down = F.conv2d(enhance_pool , self.weight_down, padding=1)

        D_left = torch.pow(D_org_letf - D_enhance_letf,2)
        D_right = torch.pow(D_org_right - D_enhance_right,2)
        D_up = torch.pow(D_org_up - D_enhance_up,2)
        D_down = torch.pow(D_org_down - D_enhance_down,2)
        E = (D_left + D_right + D_up +D_down)
        # E = 25*(D_left + D_right + D_up +D_down)

        return E



''' Optical flow warping function '''

def warp(x, flo, padding_mode='border'):
    B, C, H, W = x.size()

    # Mesh grid
    xx = torch.arange(0, W).view(1,-1).repeat(H,1)
    yy = torch.arange(0, H).view(-1,1).repeat(1,W)
    xx = xx.view(1,1,H,W).repeat(B,1,1,1)
    yy = yy.view(1,1,H,W).repeat(B,1,1,1)
    grid = torch.cat((xx,yy),1).float()
    if x.is_cuda:
        grid = grid.cuda()
    vgrid = grid - flo
    
    # Scale grid to [-1,1]
    vgrid[:,0,:,:] = 2.0*vgrid[:,0,:,:]/max(W-1,1)-1.0
    vgrid[:,1,:,:] = 2.0*vgrid[:,1,:,:]/max(H-1,1)-1.0
    vgrid = vgrid.permute(0,2,3,1)
    output = F.grid_sample(x, vgrid, padding_mode=padding_mode, mode='nearest')
    return output




''' The proposed Compound Regularization '''

class TemporalLoss(nn.Module):
    def __init__(self, data_sigma=True, data_w=True, noise_level=0.001, 
                       motion_level=8, shift_level=10):

        super(TemporalLoss,self).__init__()
        self.MSE = torch.nn.MSELoss()

        self.data_sigma = data_sigma
        self.data_w = data_w
        self.noise_level = noise_level
        self.motion_level = motion_level
        self.shift_level = shift_level

    """ Flow should have most values in the range of [-1, 1]. 
        For example, values x = -1, y = -1 is the left-top pixel of input, 
        and values  x = 1, y = 1 is the right-bottom pixel of input.
        Flow should be from pre_frame to cur_frame """

    def GaussianNoise(self, ins, mean=0, stddev=0.001):
        stddev = stddev + random.random() * stddev
        noise = Variable(ins.data.new(ins.size()).normal_(mean, stddev))
        
        if ins.is_cuda:
            noise = noise.cuda()
        return ins + noise

    def GenerateFakeFlow(self, height, width):
        ''' height: img.shape[0]
            width:  img.shape[1] '''

        if self.motion_level > 0:
            flow = np.random.normal(0, scale=self.motion_level, size = [height//100, width//100, 2])
            flow = cv2.resize(flow, (width, height))
            flow[:,:,0] += random.randint(-self.shift_level, self.shift_level)
            flow[:,:,1] += random.randint(-self.shift_level, self.shift_level)
            flow = cv2.blur(flow,(100,100))
        else:
            flow = np.ones([width,height,2])
            flow[:,:,0] = random.randint(-self.shift_level, self.shift_level)
            flow[:,:,1] = random.randint(-self.shift_level, self.shift_level)

        return torch.from_numpy(flow.transpose((2, 0, 1))).float()

    def GenerateFakeData(self, first_frame):
        ''' Input should be a (H,W,3) numpy of value range [0,1]. '''

        if self.data_w:
            forward_flow = self.GenerateFakeFlow(first_frame.shape[2], first_frame.shape[3])
            if first_frame.is_cuda:
                forward_flow = forward_flow.cuda()
            forward_flow = forward_flow.expand(first_frame.shape[0], 2, first_frame.shape[2], first_frame.shape[3])
            second_frame_ori = warp(first_frame, forward_flow)
        else:
            second_frame = first_frame.clone()
            forward_flow = None

        if self.data_sigma:
            second_frame = self.GaussianNoise(second_frame_ori, stddev=self.noise_level)

        return second_frame_ori, second_frame, forward_flow

    def forward(self, first_frame, second_frame, forward_flow):
        if self.data_w:
            first_frame = warp(first_frame, forward_flow)

        temporalloss = torch.mean(torch.abs(first_frame - second_frame))
        return temporalloss, first_frame

    ############################################################
    # Additional functions for ablation study in Figure 16.
    # ----------------------------------------------------------

    def MPI_Version(self, new_cur_frame, pre_frame, backward_flow, backward_mask):
        fake_pre_frame = warp(new_cur_frame, backward_flow) * backward_mask
        pre_frame = pre_frame * backward_mask

        ## L1 Version
        temporalloss = torch.mean(torch.abs(fake_pre_frame - pre_frame))

        ## L2 Sqrt Version
        # temporalloss = self.MSE(fake_pre_frame, pre_frame) ** 0.5

        ## L2 Version
        # temporalloss = self.MSE(fake_pre_frame, pre_frame)

        return temporalloss, fake_pre_frame

    def Video_Version(self, cur_frame, pre_frame, forward_flow, forward_mask):
        fake_cur_frame = warp(pre_frame, forward_flow) * forward_mask
        cur_frame = cur_frame * forward_mask

        ## L1 Version
        temporalloss = torch.mean(torch.abs(fake_cur_frame - cur_frame))

        ## L2 Sqrt Version
        # temporalloss = self.MSE(fake_pre_frame, pre_frame) ** 0.5

        ## L2 Version
        # temporalloss = self.MSE(fake_pre_frame, pre_frame)
        
        return temporalloss, fake_cur_frame



