# -*- coding: utf-8 -*-
"""
Created on Fri Sep 27 20:16:33 2024

@author: 35518
"""

from argparse import ArgumentParser

class TestOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
        # GPU settings  ------------------------------------------------------------------------  GPU设置
		self.parser.add_argument('--cuda', action='store_true', help='enables cuda')
		self.parser.add_argument('--gpu', action='store', type=int, default=0, help='gpu device')
        
        # Dataset settings   -----------------------------------------------------------------------------------------------训练数据集，dataloader参数
		self.parser.add_argument('--content_data', default='./mpi/shaman_2/', help='path to content images')
		self.parser.add_argument('--style_data', default='./val/style1/', help='path to style images')
        
        #Definition of Network  -----------------------------------------------------------------------------------------------   网络参数设置
		self.parser.add_argument('--style_attention', action='store_true', help='use both style and content attention filter in the decoder')
		self.parser.add_argument('--use_Global', action='store_true', help='use different network')
		self.parser.add_argument('--checkpoint', default='Model/output/result_recon.pth', type=str , help='load network path')
        
        #Output setting   -----------------------------------------------------------------------------------------------   输出设置
		self.parser.add_argument('--result_frames_path', default='result_frames', help='path to save image')
		self.parser.add_argument('--result_videos_path', default='result_videos', help='path to save vedio')
        
        
	def parse(self):
		opts = self.parser.parse_args()
		return opts