import cv2
import torch

def adjust_learning_rate(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.5
    return optimizer


def transform_image(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img.div_(255.0)
    img = (img - mean) / std
    return img.unsqueeze(0)

def tensor2numpy(img):
    img = img.data.cpu()
    img = img.numpy().transpose((1, 2, 0))
    return cv2.cvtColor(img,cv2.COLOR_RGB2BGR)

def transform_back_image(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img*std+mean
    img = img.clamp(0, 1)[0,:,:,:] * 255
    return img

def numpy2tensor(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose((2, 0, 1))).float()

def RGB2Gray(image):
    mean = image.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = image.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)

    image = (image*std+mean)

    gray = image[:,2:3,:,:]*0.299 + image[:,1:2,:,:]*0.587 + image[:,0:1,:,:]*0.114
    gray = gray.expand(image.size())

    gray = (gray-mean)/std

    return gray