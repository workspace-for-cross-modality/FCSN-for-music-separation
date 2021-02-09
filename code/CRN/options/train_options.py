from .base_options import BaseOptions
from utils import utils
import torch
import os
import argparse

class TrainOptions(BaseOptions):
	def initialize(self):
		BaseOptions.initialize(self)
		self.parser.add_argument('--display_freq', type=int, default=1, help='frequency of displaying average loss and accuracy')
		self.parser.add_argument('--save_latest_freq', type=int, default=50, help='frequency of saving the latest results')
		self.parser.add_argument('--continue_train', default=False, help='continue training: load the latest model')
		self.parser.add_argument('--decay_factor', type=float, default=0.1, help='decay factor for learning rate')
		self.parser.add_argument('--tensorboard', type=bool, default=False, help='use tensorboard to visualize loss change ')
		self.parser.add_argument('--measure_time', type=bool, default=False, help='measure time of different steps during training')

		self.parser.add_argument('--num_epoch', default=300, type=int, help='number of batches to train')
		self.parser.add_argument('--num_per_mix', default=3, type=int, help='number of video clips to mix')
		self.parser.add_argument('--num_object_per_video', default=2, type=int, help='max number of objects detected in a video clip')		
		self.parser.add_argument('--validation_on', type=bool, default=False, help='whether to test on validation set during training')
		self.parser.add_argument('--validation_freq', type=int, default=200, help='frequency of testing on validation set')
		self.parser.add_argument('--validation_batches', type=int, default=20, help='number of batches to test for validation')
		self.parser.add_argument('--validation_visualization', type=bool, default=False, help='whether save validation predictions')
		self.parser.add_argument('--num_visualization_examples', type=int, default=20, help='number of examples to visualize')		
		self.parser.add_argument('--subtract_mean', default=True, type=bool, help='subtract channelwise mean from input image')
		self.parser.add_argument('--preserve_ratio', default=False, type=bool, help='whether boudingbox aspect ratio should be preserved when loading')
		self.parser.add_argument('--enable_data_augmentation', type=bool, default=True, help='whether to augment input audio/image')

		#model arguments
		self.parser.add_argument('--visual_pool', type=str, default='conv1x1', help='avg/max pool or using a conv1x1 layer for visual stream feature')
		self.parser.add_argument('--classifier_pool', type=str, default='maxpool', help="avg or max pool for classifier stream feature")

		self.parser.add_argument('--weights_visual', type=str, default='', help="weights for visual stream ")
		self.parser.add_argument('--weights_unet', type=str, default='', help="weights for unet__")
		self.parser.add_argument('--weights_classifier', type=str, default='', help="weights for audio classifier ")
		self.parser.add_argument('--weights_refine', type=str, default='', help='weights path for refine network')
		self.parser.add_argument('--weights_down_channels', type=str, default='',help='weights path for refine network')

		self.parser.add_argument('--unet_num_layers', type=int, default=7, choices=(5, 7), help="unet number of layers")
		self.parser.add_argument('--unet_ngf', type=int, default=64, help="unet base channel dimension")
		self.parser.add_argument('--unet_input_nc', type=int, default=1, help="input spectrogram number of channels")
		self.parser.add_argument('--unet_output_nc', type=int, default=1, help="output spectrogram number of channels")
		self.parser.add_argument('--number_of_classes', default=21, type=int, help='number of classes')
		self.parser.add_argument('--classifier_loss_weight', default=0.05, type=float, help='weight for classifier loss')
		self.parser.add_argument('--coseparation_loss_weight', default=1, type=float, help='weight for reconstruction loss')
		self.parser.add_argument('--refine_loss_weight', default=1, type=float, help='weight for refine_reconstruction loss')



		self.parser.add_argument('--mask_loss_type', default='L1', type=str, choices=('L1', 'L2', 'BCE'), help='type of reconstruction loss on mask')
		self.parser.add_argument('--weighted_loss', default=True, action='store_true', help="weighted loss")
		self.parser.add_argument('--log_freq', type=bool, default=True, help="whether use log-scale frequency")		
		self.parser.add_argument('--mask_thresh', default=0.5, type=float, help='mask threshold for binary mask')
		self.parser.add_argument('--with_additional_scene_image', action='store_true', help="whether to append an extra scene image")	

		#optimizer arguments
		self.parser.add_argument('--lr_visual', type=float, default=0.0001, help='learning rate for visual stream')
		self.parser.add_argument('--lr_unet', type=float, default=0.0001, help='learning rate for unet')
		self.parser.add_argument('--lr_classifier', type=float, default=0.00001, help='learning rate for audio classifier')
		self.parser.add_argument('--lr_refine', type=float, default=0.0001, help='learning rate for refine unet7 network')
		self.parser.add_argument('--lr_audio_ext', type=float, default=0.00001, help='learning rate for audio extractor')
		self.parser.add_argument('--lr_steps', nargs='+', type=int, default=[120,240], help='steps to drop LR in training samples')

		self.parser.add_argument('--optimizer', default='adam', type=str, help='adam or sgd for optimization')
		self.parser.add_argument('--beta1', default=0.5, type=float, help='momentum for sgd, beta1 for adam')
		self.parser.add_argument('--weight_decay', default=0.0001, type=float, help='weights regularizer')

		self.parser.add_argument('--model', type=str, default='audioVisualMUSIC', help='chooses how datasets are loaded. audioVisualMUSIC or Align')
		self.parser.add_argument('--audio_extractor', type=bool, default=True, help='whether to use audio extractor to down sample the feature map, for L1/L2 calculate, replace D as regression')
		self.parser.add_argument('--visual_unet_encoder', type=bool, default=False)
		self.parser.add_argument('--visual_cat', type=bool, default=True)
		self.parser.add_argument('--refine_iteration', type=int, default=3, help='0:不要refine网络;1:是原始的方法，只有一次计算;2以上:就是refine网路的迭代次数')
		self.parser.add_argument('--name', type=str, default='sup12_music21_retrain_3mix',
								 help='name of the experiment. It decides where to store models')
		self.parser.add_argument('--left_mask_punish', type=bool, default=False)
		self.parser.add_argument('--punish_weight', type=float, default=0.00005)
		self.parser.add_argument('--multisource', type=bool, default=False)
		self.parser.add_argument('--vggsound', type=bool, default=False)
		self.parser.add_argument('--audioset', type=bool, default=False)
		self.parser.add_argument('--sameclass', type=bool, default=False)
		self.parser.add_argument('--cla_for_same', type=str, default='acoustic_guitar')
		self.parser.add_argument('--factor_type', type=str, default='down', choices=('down','up','mean'))


		self.mode = 'train'

	def parse(self):
		if not self.initialized:
			self.initialize()
		self.opt = self.parser.parse_args()
		self.opt.mode = self.mode

		args = vars(self.opt)
		print('------------ Options -------------')
		for k, v in sorted(args.items()):
			print('%s: %s' % (str(k), str(v)))
		print('-------------- End ----------------')


		# save to the disk
		expr_dir = os.path.join(self.opt.checkpoints_dir, self.opt.name)
		utils.mkdirs(expr_dir)
		file_name = os.path.join(expr_dir, 'opt.txt')
		with open(file_name, 'wt') as opt_file:
			opt_file.write('------------ Options -------------\n')
			for k, v in sorted(args.items()):
				opt_file.write('%s: %s\n' % (str(k), str(v)))
			opt_file.write('-------------- End ----------------\n')
		return self.opt