import os

import torch
import cv2
import numpy as np

# Image to tensor tools

def numpy2tensor(img):
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    return torch.from_numpy(img.transpose((2, 0, 1))).float()

def transform_image(img):
    mean = img.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = img.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    img = img.div_(255.0)
    img = (img - mean) / std
    return img.unsqueeze(0)

# Tensor to image tools

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

def read_img(img_path):
    return cv2.imread(img_path)


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


class Stylization():
    def __init__(self, opts):
        

        self.opts = opts
        self.device = torch.device("cuda" if self.opts.cuda else "cpu")

        # ===== Framework =====

        if self.opts.use_Global:
            from network.style_network_global import TransformerNet
        else:
            from network.style_networks_frame import TransformerNet
        
        self.model = TransformerNet().to(self.device)
        
        self.model.load_state_dict(torch.load(self.opts.checkpoint, map_location=lambda storage, loc: storage))
        
        for param in self.model.parameters():
            param.requires_grad = False
            
    # ===== Sequence-Level Global Feature Sharing =====

    def add_Enhance_Module(self, patch):
        with torch.no_grad():
            patch = numpy2tensor(patch).to(self.device)
            self.model.add_Enhance_Module(transform_image(patch))
        torch.cuda.empty_cache()
    
    def add_Decoder(self, patch):
        #------------------------------------- 这里不能这么加
        with torch.no_grad():
            patch = numpy2tensor(patch).to(self.device)
            self.model.add_Decoder(transform_image(patch))
        torch.cuda.empty_cache()
        
    def compute_Enhance_Module(self):
        with torch.no_grad():
            self.model.compute_Enhance_Module()
        torch.cuda.empty_cache()
    
    def compute_Decoder(self):
        with torch.no_grad():
            self.model.compute_Decoder()
        torch.cuda.empty_cache()

    def clean(self):
        self.model.clean()
        torch.cuda.empty_cache()

    # ===== Style Transfer =====

    def prepare_style(self, style):
        with torch.no_grad():
            style = numpy2tensor(style).to(self.device)
            style = transform_image(style)
            self.model.generate_style_features(style)
        torch.cuda.empty_cache()
        
    def transfer(self, frame):
        with torch.no_grad():
            # Transform images into tensors
            frame = numpy2tensor(frame).to(self.device)
            frame = transform_image(frame)
            print("frame ",frame.shape)
            # Stylization
            styled_resultA, styled_resultM = self.model(frame)
            print(styled_resultA.shape, styled_resultM.shape)
            O_auxiliary = styled_resultA
            O_main = styled_resultM[:,:3,:,:]
            O_minor = styled_resultM[:,3:,:,:]

            frame_auxiliary_result = transform_back_image(O_auxiliary)
            frame_auxiliary_result = tensor2numpy(frame_auxiliary_result)

            frame_main_result = transform_back_image(O_main)
            frame_main_result = tensor2numpy(frame_main_result)

            frame_minor_result = transform_back_image(O_minor)
            frame_minor_result = tensor2numpy(frame_minor_result)

        return frame_auxiliary_result, frame_main_result, frame_minor_result
        
def M1_validation(opts):
    style_frame = os.listdir(opts.style_data)
    content_frame = os.listdir(opts.content_data)
    
    for style_img in style_frame:

        style_img = os.path.join(opts.style_data , style_img)
        if not os.path.exists(style_img):
            exit('Style image %s not exists'%(style_img))
        style = cv2.imread(style_img)
    
        # Build model
        framework = Stylization(opts)
        framework.prepare_style(style)

        # Build tools
        reshape = ReshapeTool()
        
        frame_num = len(content_frame)
        if opts.use_Global:
            print('Preparations for Sequence-Level Global Feature Sharing')
            framework.clean()
            interval = 8
            sample_sum = (frame_num-1)//interval

            #--------------------------------------
            for s in range(sample_sum):
                i = s * interval
                print('Add frame %d , %d frames in total'%(s, sample_sum))

                content_frame_i = os.path.join(opts.content_data, content_frame[i])
                input_frame = read_img(content_frame_i)
                input_frame = reshape.process(input_frame)

                framework.add_Enhance_Module(input_frame)
            
            content_frame_i = os.path.join(opts.content_data, content_frame[-1])
            input_frame = read_img(content_frame_i)
            input_frame = reshape.process(input_frame)

            framework.add_Enhance_Module(input_frame)
        
            print('Computing Enhance_Module features')
            framework.compute_Enhance_Module()

            #--------------------------------------
            for s in range(sample_sum):
                i = s * interval
                print('Add frame %d , %d frames in total'%(s, sample_sum))

                content_frame_i = os.path.join(opts.content_data, content_frame[i])
                input_frame = read_img(content_frame_i)
                input_frame = reshape.process(input_frame)

                framework.add_Decoder(input_frame)
            
            content_frame_i = os.path.join(opts.content_data, content_frame[-1])
            input_frame = read_img(content_frame_i)
            input_frame = reshape.process(input_frame)

            framework.add_Decoder(input_frame)
        
            print('Computing Enhance_Module features')
            framework.compute_Decoder()

            print('Preparations finish!')
            
        
        
        for i in range(len(content_frame)):
            content_img = content_frame[i]
            content_img = os.path.join(opts.content_data , content_img)
            if not os.path.exists(content_img):
                exit('Style image %s not exists'%(content_img))
            input_frame = cv2.imread(content_img)
    
            # Name for this testing
            style_name = (style_img.split('/')[-1]).split('.')[0]
            video_name = (content_img.split('/')[-2])
            name = 'PMBNet-' + style_name + '-' + video_name
    
            # Mkdir corresponding folders
            if not os.path.exists('{}/{}'.format(opts.result_frames_path, name)):
                os.mkdir('{}/{}'.format(opts.result_frames_path, name))
    
            # Crop the image
            H,W,C = input_frame.shape
            new_input_frame = reshape.process(input_frame)
    
            # Stylization
            frame_auxiliary_result, frame_main_result, frame_minor_result = framework.transfer(new_input_frame)
    
            # Crop the image back
            frame_auxiliary_result = frame_auxiliary_result[64:64+H,64:64+W,:]
            frame_main_result = frame_main_result[64:64+H,64:64+W,:]
            frame_minor_result = frame_minor_result[64:64+H,64:64+W,:]
    
            # Save result
            cv2.imwrite('{}/{}/{}'.format(opts.result_frames_path, name, 
                                        "auxiliary_"+content_frame[i].split('/')[-1]), frame_auxiliary_result)
            cv2.imwrite('{}/{}/{}'.format(opts.result_frames_path, name, 
                                        "main_"+content_frame[i].split('/')[-1]), frame_main_result)
            cv2.imwrite('{}/{}/{}'.format(opts.result_frames_path, name, 
                                        "minor_"+content_frame[i].split('/')[-1]), frame_minor_result)