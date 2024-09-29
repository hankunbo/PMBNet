import cv2
import glob
import os
import scipy.io as scio
import numpy as np
import random
import time

import torch
import torch.nn as nn

from framework import Stylization
from torchvision import models, transforms
from collections import namedtuple
import torch.backends.cudnn as cudnn



class ReshapeTool():
    def __init__(self):
        self.record_H = 0
        self.record_W = 0

    def process(self, img):
        H, W, C = img.shape

        if self.record_H == 0 and self.record_W == 0:
            new_H = H + 128
            if new_H % 64 != 0:
                new_H += 64 - new_H % 64

            new_W = W + 128
            if new_W % 64 != 0:
                new_W += 64 - new_W % 64

            self.record_H = new_H
            self.record_W = new_W

        new_img = cv2.copyMakeBorder(img, 64, self.record_H-64-H,
                                          64, self.record_W-64-W, cv2.BORDER_REFLECT)
        return new_img

## -------------------
##  Parameters


# Target styles
content_img = './content/avril_cropped.jpg'

# Target content video
# Use glob.glob() to search for all the frames
# Add sort them by sort()

# path = './TestStyles'
# file_list = os.listdir(path)
# for file in file_list:
# content_video = './TestStyles/*.jpg'
teststyles = './TestStyles/*.jpg'


# Path of the checkpoint (please download and replace the empty file)
checkpoint_path = "./Model/style_net-TIP-final-3.3.pth"

# Device settings, use cuda if available
cuda = torch.cuda.is_available()

# The proposed Sequence-Level Global Feature Sharing
use_Global = False
# Saving settings
save_video = False
fps = 24

# Where to save the results
result_frames_path = './result_frames3.3_LOSS/'
result_videos_path = './result_videos3.3_LOSS/'


## -------------------
##  Tools


if not os.path.exists(result_frames_path):
    os.mkdir(result_frames_path)

if not os.path.exists(result_videos_path):
    os.mkdir(result_videos_path)


def read_img(img_path):
    return cv2.imread(img_path)






## -------------------
##  Preparation


# Read style image
# if not os.path.exists(style_img):
#     exit('Style image %s not exists'%(style_img))
# style = cv2.imread(style_img)

# Read content image
if not os.path.exists(content_img):
    exit('Style image %s not exists'%(content_img))
input_frame= cv2.imread(content_img)

# Build model
framework = Stylization(checkpoint_path, cuda, use_Global)

# Read content frames
frame_list = glob.glob(teststyles)

# Name for this testing
content_name = (content_img.split('/')[-1]).split('.')[0]
style_name = (teststyles.split('/')[-2])
name = 'SeMuVST-' + style_name + '-' + content_name
if not use_Global:
    name = name + '-no-global'

# Mkdir corresponding folders
if not os.path.exists('{}/{}'.format(result_frames_path,name)):
    os.mkdir('{}/{}'.format(result_frames_path,name))

# Build tools
reshape = ReshapeTool()




## -------------------
##  Inference


frame_num = len(frame_list)

# Prepare for proposed Sequence-Level Global Feature Sharing

# if use_Global:

#     print('Preparations for Sequence-Level Global Feature Sharing')
#     framework.clean()
#     interval = 8
#     sample_sum = (frame_num-1)//interval
    
#     for s in range(sample_sum):
#         i = s * interval
#         print('Add frame %d , %d frames in total'%(s, sample_sum))
#         input_frame = read_img(frame_list[i])
#         style = cv2.imread(style_img)
#         framework.add(input_frame, style)

#     input_frame = read_img(frame_list[-1])
#     framework.add(input_frame, style)

#     print('Computing global features')
#     framework.compute()

#     print('Preparations finish!')

    # set device on GPU if available, else CPU
if cuda:
    cudnn.benchmark = True
    device = torch.device(0)

else:
    device = 'cpu'

vgg_outputs = namedtuple("VggOutputs", ['relu1_1', 'relu2_1', 'relu3_1', 'relu4_1'])

class Vgg19(nn.Module):
    def __init__(self, cuda = True, requires_grad=False):
        super(Vgg19, self).__init__()
        if cuda:
            cudnn.benchmark = True
            self.device = torch.device(0)
        else:
            self.device = torch.device("cpu")
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


Vgg19 = Vgg19().to(device)
MSE = nn.MSELoss()
def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std
# Main stylization

def style_loss(features_coded_Image, features_style):
    style_loss = 0.
    for ft_x, ft_s in zip(features_coded_Image, features_style):
        mean_x, var_x = calc_mean_std(ft_x)
        mean_style, var_style = calc_mean_std(ft_s)

        style_loss = style_loss + MSE(mean_x, mean_style)
        style_loss = style_loss + MSE(var_x, var_style)

    return style_loss


def content_loss(features_coded_Image, features_content):
    content_loss = MSE(features_coded_Image.relu4_1, features_content.relu4_1)
    return content_loss

def numpy2tensor(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose((2, 0, 1))).float()

def transform_image(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img.div_(255.0)
    img = (img - mean) / std
    return img.unsqueeze(0)

start  = time.time()
content_Loss = 0
style_Loss = 0
# Crop the image
H,W,C = input_frame.shape
new_input_frame = reshape.process(input_frame)
input_frame = numpy2tensor(input_frame).to(device)
input_frame = transform_image(input_frame)
for i in range(frame_num):

    print("Stylizing frame %d"%(i))

    # Read the image
    style = read_img(frame_list[i])
    framework.prepare_style(style)


    # Crop the image
    #H,W,C = input_frame.shape
    # new_input_frame = reshape.process(input_frame)

    # Stylization
    styled_input_frame = framework.transfer(new_input_frame)




    # Crop the image back
    styled_input_frame = styled_input_frame[64:64+H,64:64+W,:]

    h,w,c = styled_input_frame.shape
    # style = cv2.resize(style,(h,w))
    style = numpy2tensor(style).to(device)
    style = transform_image(style)
    styled_frame = numpy2tensor(styled_input_frame).to(device)
    styled_frame = transform_image(styled_frame)



    F_styled = Vgg19(styled_frame)
    F_style_gt = Vgg19(style)
    F_content_gt = Vgg19(input_frame)
    ori_style_loss = style_loss(F_styled, F_style_gt)
    style_Loss = style_Loss + ori_style_loss
    contentloss = content_loss(F_styled, F_content_gt)
    content_Loss = content_Loss + contentloss

    # Save result
    cv2.imwrite('{}/{}/{}'.format(result_frames_path, name, 
                                frame_list[i].split('/')[-1]), styled_input_frame)


print(content_Loss)
print(style_Loss)


    

                               
# end = time.time()
# running_time = end-start
# print('time cost: %.5f sec' %running_time)

# Write images back to video

# if save_video:
#     frame_list = glob.glob("{}/{}/*.*".format(result_frames_path,name))
#     frame_list.sort()
#     demo = cv2.imread(frame_list[0])
    
#     fourcc = cv2.VideoWriter_fourcc(*'MJPG') # Also (*'DVIX') or (*'X264')
#     videoWriter = cv2.VideoWriter('{}/{}.avi'.format(result_videos_path, name), 
#                                 fourcc, fps, (demo.shape[1],demo.shape[0]))

#     for frame in frame_list:
#         videoWriter.write(cv2.imread(frame))
#     videoWriter.release()
