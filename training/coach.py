import os
import cv2
import random

import torch
import torch.nn as nn
import torch.optim as optim
import glob
import numpy as np

from network.style_networks import TransformerNet
from network.other_networks import define_D, init_weights, GANLoss

from training.tools import adjust_learning_rate, transform_image, tensor2numpy, transform_back_image, numpy2tensor, RGB2Gray

from training.dataset import get_loader

from loss.new_loss import L_spa, L_exp
from loss.ssim_loss import SSIM
from loss.loss_networks import warp, TemporalLoss
from loss.loss_function import crt_loss, style_total_loss, content_loss, Lp_loss

class Validation():
    def __init__(self, opts):
        self.opts = opts
        self.device = torch.device("cuda" if self.opts.cuda else "cpu")
        content_img_list = glob.glob(self.opts.valf+'/content/*.jpg')
        self.content_image = []
        for i in range(6):
            img = cv2.imread(content_img_list[i])
            img = cv2.resize(img, (256,256))
            img = numpy2tensor(img).to(self.device)
            img = transform_image(img)
            self.content_image.append(img)
            
            

        style_img_list = glob.glob(self.opts.valf+'/style/*.jpg')
        self.style_image = []
        for i in range(6):
            img = cv2.imread(style_img_list[i])
            img = cv2.resize(img, (256,256))
            img = numpy2tensor(img).to(self.device)
            img = transform_image(img)
            self.style_image.append(img)

    def SaveResults(self, style_net, epoch):
        with torch.no_grad():
            for i in range(6):
                content = self.content_image[i].to(self.device)
                style = self.style_image[i].to(self.device)
                
                O_auxiliary, O_main, O_minor = style_net.validation(content, style)
                '''
                styled_resultA, styled_resultM, \
                F_styledA, F_styledM, F_content_gt, F_style_gt,\
                recon_contentA, recon_styleA, recon_contentM, recon_styleM = style_net(content, style)
                
                O_auxiliary = styled_resultA
                O_main = styled_resultM[:,:3,:,:]
                O_minor = styled_resultM[:,3:,:,:]
                '''
                frame_auxiliary_result = transform_back_image(O_auxiliary)
                frame_auxiliary_result = tensor2numpy(frame_auxiliary_result)
    
                frame_main_result = transform_back_image(O_main)
                frame_main_result = tensor2numpy(frame_main_result)
    
                frame_minor_result = transform_back_image(O_minor)
                frame_minor_result = tensor2numpy(frame_minor_result)
                
                # Save result
                content = transform_back_image(content)
                content = tensor2numpy(content)
                cv2.imwrite('{}/Epoch[{}]-validation_content-{}.png'.format(self.opts.outf, epoch, i), content)
                cv2.imwrite('{}/Epoch[{}]-validation_auxiliary-{}.png'.format(self.opts.outf, epoch, i), frame_auxiliary_result)
                cv2.imwrite('{}/Epoch[{}]-validation_main-{}.png'.format(self.opts.outf, epoch, i), frame_main_result)
                cv2.imwrite('{}/Epoch[{}]-validation_minor-{}.png'.format(self.opts.outf, epoch, i), frame_minor_result)
                
class TV(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TV,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h =  (x.size()[2]-1) * x.size()[3]

        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

class Coach:
    def __init__(self, opts):
        self.opts = opts
        
        if self.opts.manualSeed is None:
            self.opts.manualSeed = random.randint(1, 10000)
        print("Random Seed: ", self.opts.manualSeed)
        random.seed(self.opts.manualSeed)
        torch.manual_seed(self.opts.manualSeed)
        
        if self.opts.cuda:
            torch.cuda.manual_seed_all(self.opts.manualSeed)
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            
        # Build model
        self.style_net = TransformerNet(style_attention=self.opts.style_attention)
        if self.opts.continue_training:
            self.style_net.load_state_dict(torch.load(self.opts.pre_model, map_location=lambda storage, loc: storage), strict=False)
            
        # Set states
        self.style_net.train()
        
        # Set devices
        os.environ["CUDA_VISIBLE_DEVICES"] = "%d"%(self.opts.gpu)
        self.device = torch.device("cuda" if self.opts.cuda else "cpu")
        self.style_net = self.style_net.to(self.device)
        
        # ===== Optimizer =====
        self.optimizer = optim.Adam(filter(lambda p: p.requires_grad, self.style_net.parameters()), lr=self.opts.lr)
        
        # Savepath
        if not os.path.exists(self.opts.outf):
            os.mkdir(self.opts.outf)
        
        self.Validation = Validation(self.opts)
        self.TV = TV()
        
        #loss function
        if self.opts.spa_loss:
            self.L_spa = L_spa()
            
        if self.opts.exp_loss:
            self.L_exp = L_exp(16,0.6)
            
        if self.opts.ssim_loss:
            self.SSIM = SSIM()

        if self.opts.adaversarial_loss:
            self.netD = define_D(3).to(self.device)
            init_weights(self.netD)
            self.netD.train()
            
            if self.opts.continue_training:
                checkpoint = self.opts.pre_modelD
                if os.path.exists(checkpoint):
                    self.netD.load_state_dict(torch.load(checkpoint, map_location=lambda storage, loc: storage))

            self.criterionGAN = GANLoss('lsgan').to(self.device)
            self.optimizerD = optim.Adam(filter(lambda p: p.requires_grad, self.netD.parameters()), lr=1e-4, betas=(0.5, 0.9))
            
        self.loader = get_loader(self.opts.batchSize, 
                                loadSize=self.opts.loadSize, 
                                fineSize=self.opts.fineSize, 
                                flip=self.opts.flip,
                                content_path=self.opts.content_data, 
                                style_path=self.opts.style_data, 
                                num_workers=self.opts.num_workers, 
                                use_mpi=False, 
                                use_video=False)
        print('Data Load Success.')
        self.iteration_sum = len(self.loader)
        
        if self.opts.temporal_loss:
            self.TemporalLoss = TemporalLoss(data_sigma = self.opts.data_sigma, 
                                             data_w = self.opts.data_w, 
                                             noise_level = self.opts.data_noise_level,
                                             motion_level = self.opts.data_motion_level,
                                             shift_level = self.opts.data_shift_level
                                             )
        if self.opts.spa_loss:
            self.L_spa = L_spa()
        if self.opts.exp_loss:
            self.L_exp = L_exp(16,0.6)
        if self.opts.ssim_loss:
            self.SSIM = SSIM()
        
        
    def Validate(self, epoch):
        self.Validation.SaveResults(self.style_net, epoch)
        
    def train_NetD(self, content, style):
        Content = content.to(self.device)
        Style = style.to(self.device)
        
        # 1. Change state of networks
        for param in self.netD.parameters():
            param.requires_grad = True
        self.style_net.ParamStatic()

        self.optimizerD.zero_grad()

        # 2. Output of netG
        with torch.no_grad():
            StyledFirstFrame, _, _ = self.style_net.validation(Content, Style)

        # 3. Train netD
        # Fake data
        pred_fake = self.netD(StyledFirstFrame.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)

        # Real data
        pred_real = self.netD(Style)
        loss_D_real = self.criterionGAN(pred_real, True)

        # Combine loss and calculate gradients
        loss_D = (loss_D_fake + loss_D_real) * 0.5

        loss_D.backward()
        self.optimizerD.step()
        
    def train_DecM(self, content, style):
        Content = content.to(self.device)
        Style = style.to(self.device)
        
        self.optimizer.zero_grad()
        self.style_net.Train_DM()
        
        styled_resultA, styled_resultM, \
        F_styledA, F_styledM, F_content_gt, F_style_gt,\
        recon_contentA, recon_styleA, recon_contentM, recon_styleM = self.style_net(Content, Style)
        
        Loss1 = 0.
        L_ctr = content_lossM = new_style_lossM = lp_loss = recon_lossM = temporal_loss = 0.
        if self.opts.with_IRT:
            L_ctr =crt_loss(styled_resultM, styled_resultA, Content)
            Loss1 += L_ctr * self.opts.IRTWeight
        
        if self.opts.style_content_loss:
            content_lossM = content_loss(F_styledM, F_content_gt)
            robust_tmp_styleM, new_style_lossM, ori_style_lossM = style_total_loss(Style, F_styledM, F_style_gt)
            Loss1 += content_lossM * self.opts.contentWeight + new_style_lossM * self.opts.styleWeight
        
        if self.opts.with_LP:
            lp_loss = Lp_loss(styled_resultA, styled_resultM[:,:3,:,:])
            Loss1 += lp_loss * self.opts.LPWeight
        
        ## Reconstruction loss
        if self.opts.recon_loss:
            recon_lossM = torch.mean(torch.abs(recon_contentM[:,:3,:,:] - Content)) + torch.mean(torch.abs(recon_styleM[:,:3,:,:] - Style))
            recon_ssim_lossM = 2-self.SSIM(recon_contentM[:,:3,:,:],Content)-self.SSIM(recon_styleM[:,:3,:,:],Style)
            Loss1 = Loss1 + recon_lossM * self.opts.reconWeight
            
        # Update
        Loss1.backward()
        self.optimizer.step()
        
        return new_style_lossM, content_lossM, L_ctr, Loss1
        
    def train_DecA(self, content, style):
        Content = content.to(self.device)
        Style = style.to(self.device)
        
        self.optimizer.zero_grad()
        self.style_net.Train_DA()
        
        styled_resultA, styled_resultM, \
        F_styledA, F_styledM, F_content_gt, F_style_gt,\
        recon_contentA, recon_styleA, recon_contentM, recon_styleM = self.style_net(Content, Style)
        
        Loss2 = 0.
        tv_loss = loss_G_GAN = ssim_loss = spa_loss = exp_loss =  temporal_loss = temporal_loss_GT = \
        recon_lossA = recon_ssim_lossA = content_lossA = new_style_lossA = ori_style_lossA = 0.
        
        if self.opts.temporal_loss:
            FirstFrame = Content
            SecondFrame_ori, SecondFrame, ForwardFlow = self.TemporalLoss.GenerateFakeData(FirstFrame)
            StyledSecondFrame , _, _ = self.style_net.validation(SecondFrame, Style)

            temporal_loss, FakeStyledSecondFrame_1 = self.TemporalLoss(styled_resultA, StyledSecondFrame, ForwardFlow)

            temporal_loss_GT, _ = self.TemporalLoss(FirstFrame, SecondFrame, ForwardFlow)
            
            if self.opts.ssim_loss:
                res = SecondFrame_ori - FirstFrame
                res = RGB2Gray(res)
                res_s = StyledSecondFrame - styled_resultA
                res_s = RGB2Gray(res_s)
                RES = res - res_s
                ssim_loss = 1-self.SSIM(res, res_s) + self.TV(RES)

            Loss2 = Loss2 + temporal_loss * self.opts.temporalWeight + ssim_loss * self.opts.ssimWeight
            
        if self.opts.spa_loss:
            spa_loss = torch.mean(self.L_spa(styled_resultA, FirstFrame))
            Loss2 = Loss2 + spa_loss * self.opts.spaWeight  
        
        if self.opts.exp_loss:
            exp_loss = torch.mean(self.L_exp(styled_resultA))
            Loss2 = Loss2 + exp_loss * self.opts.expWeight  
            
        if self.opts.recon_loss:
            recon_lossA = torch.mean(torch.abs(recon_contentA-Content)) + torch.mean(torch.abs(recon_styleA-Style))
            Loss2 = Loss2 + recon_lossA * self.opts.reconWeight
            
        if self.opts.recon_ssim_loss:
            recon_ssim_lossA = 2-self.SSIM(recon_contentA,Content)-self.SSIM(recon_styleA,Style)
            Loss2 = Loss2 + recon_ssim_lossA * self.opts.recon_ssimWeight
        
        if self.opts.style_content_loss:
            content_lossA = content_loss(F_styledA, F_content_gt)
            robust_tmp_styleA, new_style_lossA, ori_style_lossA = style_total_loss(Style, F_styledA, F_style_gt)
            Loss2 = Loss2 + content_lossA * self.opts.contentWeight + new_style_lossA * self.opts.styleWeight
            
        if self.opts.tv_loss:
            tv_loss = self.TV(styled_resultA)
            Loss2 = Loss2 + tv_loss * self.opts.tvWeight

        if self.opts.old_style_loss:
            Loss2 = Loss2 + ori_style_lossA * self.opts.oldWeight

        if self.opts.adaversarial_loss:
            pred_fake = self.netD(styled_resultA)
            loss_G_GAN = self.criterionGAN(pred_fake, True)
            Loss2 = Loss2 + loss_G_GAN * self.opts.ganWeight

        # Update
        Loss2.backward()
        self.optimizer.step()
        
        return  new_style_lossA, content_lossA, ori_style_lossA, recon_lossA, recon_ssim_lossA, ssim_loss, tv_loss, temporal_loss, temporal_loss_GT, loss_G_GAN, Loss2
    
    def Train(self):
        min_total_loss = np.inf
        cur_total_loss = 0.
        for epoch in range(self.opts.load_epoch+1, self.opts.epoches+1):
    
            for iteration, sequence in enumerate(self.loader):
                
                Content = sequence['Content'].to(self.device)
                Style = sequence['Style'].to(self.device)
                #------------------------------------
                #train netD
                #------------------------------------
                if self.opts.adaversarial_loss:
                    self.train_NetD(Content, Style)
                    
                #------------------------------------
                #train decoder main
                #------------------------------------
                new_style_lossM, content_lossM, L_ctr, Loss1 = self.train_DecM(Content, Style)
    
                if iteration % 10 == 0:
                    print("[Epoch %d/%d][Iter %d/%d] New Style: %.3f, Content: %.3f, CRT: %.3f, TOTAL: %.3f" \
                    % (epoch, self.opts.epoches, iteration, self.iteration_sum, new_style_lossM, content_lossM, L_ctr, Loss1)) 
                        
                #------------------------------------
                #train decoder auxiliary
                #------------------------------------
                new_style_lossA, content_lossA, ori_style_lossA, recon_lossA, recon_ssim_lossA, ssim_loss, tv_loss, temporal_loss, temporal_loss_GT, loss_G_GAN, Loss2 = self.train_DecA(Content, Style)
                
                cur_total_loss += Loss2.item()
        
                if iteration % 10 == 0:
    
                    print("[Epoch %d/%d][Iter %d/%d] New Style: %.3f, Content: %.3f, Old Style: %.3f, Recon: %.3f, ReconSSIM: %.3f, SSIM: %.3f, TV: %.3f, Temporal: %.3f (%.3f), GAN: %.3f, TOTAL: %.3f" \
                    % (epoch, self.opts.epoches, iteration, self.iteration_sum, new_style_lossA, content_lossA, ori_style_lossA, recon_lossA, recon_ssim_lossA, ssim_loss, tv_loss, temporal_loss, temporal_loss_GT, loss_G_GAN, Loss2))
                        
                        
                if iteration % self.opts.log == 0 and iteration != 0:
                    cur_total_loss /= self.opts.log
                    print("cur_total_loss:", cur_total_loss, "min_total_loss: ", min_total_loss)
                    if cur_total_loss < min_total_loss:
                        min_total_loss = cur_total_loss
                        torch.save(self.optimizer.state_dict(), '%s/optimizer-epoch-%d.pth' % (self.opts.outf, epoch))
                        torch.save(self.style_net.state_dict(), '%s/style_net-latest-epoch-%d-%d.pth' % (self.opts.outf, epoch, iteration))
    
                        if self.opts.adaversarial_loss:
                            torch.save(self.netD.state_dict(), '%s/netD-epoch-%d.pth' % (self.opts.outf, epoch))
                    cur_total_loss = 0
    
                    self.Validation.SaveResults(self.style_net, epoch)
                    
            
            
