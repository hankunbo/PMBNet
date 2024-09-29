from argparse import ArgumentParser

class TrainOptions:

	def __init__(self):
		self.parser = ArgumentParser()
		self.initialize()

	def initialize(self):
        # GPU settings  ------------------------------------------------------------------------  GPU设置
		self.parser.add_argument('--cuda', action='store_true', help='enables cuda')
		self.parser.add_argument('--gpu', action='store', type=int, default=0, help='gpu device')
		self.parser.add_argument('--manualSeed', type=int, help='manual seed')
        
        # Save path settings  ------------------------------------------------------------------------  保存路径设置
		self.parser.add_argument('--outf', default='result', help='path to output images and model checkpoints')
		self.parser.add_argument('--valf', default='val', help='path to validation images')
        
		'''
        # Basic training settings
		
		
		self.parser.add_argument('--start_iteration', type=int, default=0, help='can start at any iteration')
		


		self.parser.add_argument('--log_dir', default='log', help='path to event file of tensorboard')
        
        # Data augmentation settings
		
		self.parser.add_argument('--train_only_decoder', action='store_true', help='both content and style encoder are fixed pre-trained VGG')

		# Specific settings for Compound Regularization (proposed model: --data_sigma --data_w)
		
        '''
        
        # 建议最后再整理一下， 1.删除没用过的， 2.合并同类型的
        # Dataset settings   -----------------------------------------------------------------------------------------------训练数据集，dataloader参数
		self.parser.add_argument('--content_data', default='./data/content/', help='path to content images')
		self.parser.add_argument('--style_data', default='./data/style/', help='path to style images')
		self.parser.add_argument('--batchSize', type=int, default=8, help='input batch size')
		self.parser.add_argument('--loadSize', type=int, default=512, help='the height / width of the input image to network')
		self.parser.add_argument('--fineSize', type=int, default=256, help='the height / width of the input image to network')
		self.parser.add_argument('--flip', type=int, default=1, help='1 for flipping image randomly, 0 for not')
		self.parser.add_argument('--num_workers', type=int, default=4, help='num of workers for data loader')
        
        #Definition of Network  -----------------------------------------------------------------------------------------------   网络参数设置
		self.parser.add_argument('--style_attention', action='store_true', help='use both style and content attention filter in the decoder')
        
        # data loading   -----------------------------------------------------------------------------------------------   是否采用预训练模型，预训练模型参数
		self.parser.add_argument("--continue_training", action='store_true', help="pretrained_train  是否采用预训练模型")
		self.parser.add_argument("--pre_model", default='pretrained_model/style_net-latest-epoch-1-10000.pth', type=str, help="pretrained_model  预训练模型")
		self.parser.add_argument("--pre_modelD", default=None, type=str, help="pretrained_model  预训练模型")
        
        # Basic training settings  -----------------------------------------------------------------------------------------------  基础训练设置
		self.parser.add_argument('--epoches', type=int, default=5, help='number of epoches to train for')
		self.parser.add_argument('--load_epoch', type=int, default=0, help='epoch of loaded checkpoint')
		self.parser.add_argument('--lr', type=float, default=1e-4, help='learning rate')
		self.parser.add_argument('--log', type=int, default=1000, help='number of iteration to save checkpints and figures')
        
        # loss parameter  -----------------------------------------------------------------------------------------------  损失函数参数 
		self.parser.add_argument('--temporal_loss', action='store_true', help='use temporal loss')
		self.parser.add_argument('--temporalWeight', type=float, default=60, help='temporal loss weight')
		self.parser.add_argument('--data_sigma', action='store_true', help='use noise in temporal loss')
		self.parser.add_argument('--data_w', action='store_true', help='use warp in temporal loss')
		self.parser.add_argument('--data_noise_level', type=float, default=0.001, help='noise level in temporal loss')
		self.parser.add_argument('--data_motion_level', type=float, default=8, help='motion level in temporal loss')
		self.parser.add_argument('--data_shift_level', type=float, default=10, help='shift level in temporal loss')
        
		self.parser.add_argument('--ssim_loss', action='store_true', help='use the self ssim')
		self.parser.add_argument('--ssimWeight', type=float, default=60, help='temporal loss weight')
        
		self.parser.add_argument('--style_content_loss', action='store_true', help='use style loss and content loss')
		self.parser.add_argument('--contentWeight', type=float, default=1, help='content loss weight')
		self.parser.add_argument('--styleWeight', type=float, default=20, help='style loss weight')
        
		self.parser.add_argument('--recon_loss', action='store_true', help='use reconstruction loss')
		self.parser.add_argument('--reconWeight', type=float, default=20, help='reconstruction loss weight')

		self.parser.add_argument('--spa_loss', action='store_true', help='use the spatial consistency loss')
		self.parser.add_argument('--spaWeight', type=float, default=60, help='spa loss weight')
        
		self.parser.add_argument('--exp_loss', action='store_true', help='use the Exposure Control loss')
		'''Zero-DCE spatial consistency loss Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement'''
		self.parser.add_argument('--expWeight', type=float, default=60, help='exp loss weight')
        
		self.parser.add_argument('--recon_ssim_loss', action='store_true', help='use the recon ssim')
		'''Preserving Global and Local Temporal Consistency for Arbitrary Video Style Transfer Self-similarity SSIM'''
		self.parser.add_argument('--recon_ssimWeight', type=float, default=20, help='exp loss weight')
        
		self.parser.add_argument('--tv_loss', action='store_true', help='use tv loss')
		self.parser.add_argument('--tvWeight', type=float, default=20, help='tv loss weight')
        
		self.parser.add_argument('--old_style_loss', action='store_true', help='use the classical style loss')
		self.parser.add_argument('--oldWeight', type=float, default=10, help='the classical style loss weight')

		self.parser.add_argument('--adaversarial_loss', action='store_true', help='use LSGAN (not included in the paper)')
		self.parser.add_argument('--ganWeight', type=float, default=1, help='LSGAN loss weight')
		''' Adding LSGAN can make the stylization effect better.
        The color can be more balanced, and textures can be more vivid.
        However, temporal consistency will be worse.
        So I didn't include the LSGAN loss in the paper.
        If you are interested, try adding --adaversarial_loss ヾ(=･ω･=)o '''
        
		self.parser.add_argument("--with_IRT", action='store_true', help="use IRT or not")
		self.parser.add_argument('--IRTWeight', type=float, default=20, help='tv loss weight')
        
		self.parser.add_argument("--with_LP", action='store_true', help="use LP or not")
		self.parser.add_argument('--LPWeight', type=float, default=1, help='tv loss weight')
        
        

	def parse(self):
		opts = self.parser.parse_args()
		return opts

