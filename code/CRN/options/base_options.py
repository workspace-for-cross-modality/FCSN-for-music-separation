import argparse
import os
import torch
from utils import utils

class BaseOptions():
	def __init__(self):
		self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
		self.initialized = False

	def initialize(self):
		self.parser.add_argument('--data_path', default='', help='path to frame/audio/detections')
		self.parser.add_argument('--data_path_duet', default='', help='path to frame/audio/detections')
	
		self.parser.add_argument('--hdf5_path', default='')
		self.parser.add_argument('--hri_change', default='', type=str, help='/home/ for hri-4, /data/ for hri-3')
		
		self.parser.add_argument('--num_gpus', type=int, default=2, help='gpu ids: e.g. 0  0,1,2, 0,2. use -1 for CPU')

		self.parser.add_argument('--checkpoints_dir', type=str, default='checkpoints/', help='models are saved here')

		self.parser.add_argument('--batch_size_per_gpu', type=int, default=8, help='input batch size')
		self.parser.add_argument('--nThreads', default=16, type=int, help='# threads for loading data')
		self.parser.add_argument('--seed', default=42, type=int, help='random seed')

		#audio arguments
		self.parser.add_argument('--audio_window', default=65535, type=int, help='audio segment length')
		self.parser.add_argument('--audio_sampling_rate', default=11025, type=int, help='sound sampling rate')
		self.parser.add_argument('--stft_frame', default=1022, type=int, help="stft frame length")
		self.parser.add_argument('--stft_hop', default=256, type=int, help="stft hop length")
		self.parser.add_argument('--max_audio_length', default=111463, type=int, help="max audio length")
		self.parser.add_argument('--mean_audio_length', default=110250, type=int, help="mean audio length")


		self.initialized = True


