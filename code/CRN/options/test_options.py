from .base_options import BaseOptions
import os
from utils import utils

#test by mix and separate two videos
class TestOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--video1_name', type=str)
		self.parser.add_argument('--video2_name', type=str)
		self.parser.add_argument('--video3_name', type=str)
		self.parser.add_argument('--output_dir_root', type=str, default='output')
		self.parser.add_argument('--hop_size', default=0.05, type=float, help='the hop length to perform audio separation in a sliding window approach')
		self.parser.add_argument('--subtract_mean', default=True, type=bool, help='subtract channelwise mean from input image')
		self.parser.add_argument('--preserve_ratio', default=False, type=bool, help='whether boudingbox aspect ratio should be preserved when loading')
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=False, help='whether to augment input audio/image')
		self.parser.add_argument('--spectrogram_type', type=str, default='magonly', choices=('complex', 'magonly'), help='whether to use magonly or complex spectrogram')
		self.parser.add_argument('--with_discriminator', action='store_true', help='whether to use discriminator')
		self.parser.add_argument('--visualize_spectrogram', action='store_true', help='whether to use discriminator')

		#model specification
		self.parser.add_argument('--visual_pool', type=str, default='conv1x1', help='avg or max pool for visual stream feature')
		self.parser.add_argument('--classifier_pool', type=str, default='maxpool', help="avg or max pool for classifier stream feature")
		self.parser.add_argument('--weights_visual', type=str,
								 default='', help="weights for visual stream")
		self.parser.add_argument('--weights_unet', type=str,
								 default='', help="weights for unet")
		self.parser.add_argument('--weights_classifier', type=str,
								 default='', help="weights for audio classifier")
		self.parser.add_argument('--unet_num_layers', type=int, default=7, choices=(5, 7), help="unet number of layers")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=1, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=1, help="output spectrogram number of channels")
		self.parser.add_argument('--number_of_classes', default=15, type=int, help='number of classes')
		self.parser.add_argument('--with_silence_category', action='store_true', help='whether to augment input audio/image')	
		self.parser.add_argument('--weighted_loss', action='store_true', help="weighted loss")
		self.parser.add_argument('--binary_mask', action='store_true', help="whether use binary mask, ratio mask is used otherwise")
		self.parser.add_argument('--full_frame', action='store_true', help="pass full frame instead of object regions")
		self.parser.add_argument('--mask_thresh', default=0.5, type=float, help='mask threshold for binary mask')
		self.parser.add_argument('--log_freq', type=bool, default=True, help="whether use log-scale frequency")		
		self.parser.add_argument('--with_frame_feature', action='store_true', help="whether also use frame-level visual feature")	
		self.parser.add_argument('--with_additional_scene_image', action='store_true', help="whether append an extra scene image")
		self.parser.add_argument('--num_of_object_detections_to_use', type=int, default=3, help="num of predictions to avg")
		self.parser.add_argument('--num_per_mix', default=2, type=int, help='number of video clips to mix')
		self.parser.add_argument('--epoch_for_classifier', default=1, type=int, help='epoch for test classifier')
		self.parser.add_argument('--model', type=str, default='AV_shuffle',
								 help='chooses how datasets are loaded.')
		self.parser.add_argument('--weights_refine', type=str,
								 default='', help='weights path for refine network')
		self.parser.add_argument('--audio_extractor', type=bool, default=True,
								 help='是否使用音频特征提取来对特征图进行降维计算，用于L2、L1loss计算，替代D的结构，改为回归')
		self.parser.add_argument('--weights_down_channels', type=str,
								 default='',
								 help='weights path for audio_extractor_down_channels')
		self.parser.add_argument('--visual_unet_encoder', type=bool, default=False)
		self.parser.add_argument('--visual_cat', type=bool, default=True)
		self.parser.add_argument('--refine_iteration', type=int, default=3,
								 help='0:不要refine网络;1:是原始的方法，只有一次计算;2以上:就是refine网路的迭代次数')
		self.parser.add_argument('--factor_type', type=str, default='down', choices=('down', 'up', 'mean'))
		self.parser.add_argument('--test_name', type=str, default='test_sup14_iteration_viz_5',help='test name')

		#include test related hyper parameters here
		self.mode = "test"

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.mode = self.mode


		#I should process the opt here, like gpu ids, etc.
		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		expr_dir = os.path.join('.output', self.opt.test_name)
		utils.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt
