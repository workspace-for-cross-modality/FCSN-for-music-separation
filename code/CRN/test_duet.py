#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import librosa
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import torch
import torch.nn.functional as F
from options.test_options import TestOptions
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from data.audioVisual_dataset import generate_spectrogram_magphase
from mir_eval.separation import bss_eval_sources, validate
from utils import utils
import h5py
import random
import csv
from utils.utils import warpgrid
from data.audioVisual_soloduet import sample_object_detections
from data.audioVisual_dataset import filter_det_bbs
from itertools import combinations

def clip_audio(audio):
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def get_separated_audio(outputs, batch_data, opt):
	# fetch data and predictions
	mag_mix = batch_data['audio_mix_mags']
	phase_mix = batch_data['audio_mix_phases']
	pred_masks_ = outputs['pred_mask']
	refine_mask_ = outputs['refine_mask']
	# unwarp log scale
	B = mag_mix.size(0)
	if opt.log_freq:
		grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame//2+1, pred_masks_.size(3), warp=False)).to(opt.device)
		pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
		refine_mask = F.grid_sample(refine_mask_, grid_unwarp)
	else:
		pred_masks_linear = pred_masks_
	# convert into numpy
	mag_mix = mag_mix.detach().cpu().numpy()
	phase_mix = phase_mix.detach().cpu().numpy()

	pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
	preds_wav = []
	for i in range(B):
		pred_mag = mag_mix[i, 0] * pred_masks_linear[i, 0]
		preds_wav.append(utils.istft_reconstruction(pred_mag, phase_mix[i, 0], hop_length=opt.stft_hop, length=opt.audio_window))

	refine_mask = refine_mask.detach().cpu().numpy()
	refine_wav = []
	for i in range(B):
		refine_mag = mag_mix[i,0] * refine_mask[i,0]
		refine_wav.append(utils.istft_reconstruction(refine_mag, phase_mix[i,0],hop_length=opt.stft_hop,length=opt.audio_window))
	return preds_wav, refine_wav

def getSeparationMetrics(reference_sources, estimated_sources):
        # reference_sources = np.concatenate((np.expand_dims(audio1_gt, axis=0), np.expand_dims(audio2_gt, axis=0)), axis=0)
        # #print reference_sources.shape
        # estimated_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
        #print estimated_sources.shape

        (sdr, sir, sar, perm) = bss_eval_sources(np.asarray(reference_sources), np.asarray(estimated_sources), False)
        #print sdr, sir, sar, perm
        return np.mean(sdr), np.mean(sir), np.mean(sar)

def test_sepration(opt, nets, output_dir, save_files=False):
	#load test arguments

	# altered_visual1 = opt.video1_name               #  '1oz3h9doX_g_5'
	# altered_visual2 = opt.video2_name               # '2R12lQszz90_4'
	opt.visualize_spectrogram = True

	model = AudioVisualModel(nets, opt)

	#model = torch.nn.DataParallel(model, device_ids=[0])
	model.to('cuda')
	model.eval()

	#load the two audios
	# audio1_path = os.path.join(opt.data_path, 'solo_audio_resample', opt.video1_ins, opt.video1_name + '.wav')
	# audio1, _ = librosa.load(audio1_path, sr=opt.audio_sampling_rate)
	audio2_path = os.path.join(opt.data_path_duet, 'duet_audio_resample', opt.video2_ins, opt.video2_name + '.wav')
	audio2, _ = librosa.load(audio2_path, sr=opt.audio_sampling_rate)


	#make sure the two audios are of the same length and then mix them
	# audio_length = min(len(audio1), len(audio2))
	# audio1 = clip_audio(audio1[:audio_length])
	audio_length = len(audio2)
	audio2 = clip_audio(audio2[:audio_length])
	audio_mix = audio2

	#define the transformation to perform on visual frames
	vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
	vision_transform_list_for_unet = [transforms.Resize((256, 256)), transforms.ToTensor()]
	if opt.subtract_mean:
		vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
		vision_transform_list_for_unet.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

	vision_transform = transforms.Compose(vision_transform_list)
	vision_transform_for_unet = transforms.Compose(vision_transform_list_for_unet)

	#load the object regions of the highest confidence score for both videos
	# detectionResult1 = np.load(os.path.join(opt.data_path, 'solo_detect', opt.video1_ins, opt.video1_name + '.npy'))
	detectionR_npy = os.path.join(opt.data_path_duet, 'duet_detect', opt.video2_ins, opt.video2_name + '.npy')
	detectionResult2 = np.load(detectionR_npy)
	# detectionResult3 = np.load(os.path.join(opt.data_path, 'solo_detect', opt.video3_ins, opt.video3_name + '.npy'))
	# clip_det_bbs1 = sample_object_detections(detectionResult1)
	clip_2_path = os.path.join('/data/mashuo/work/study/refine-separation/dataset/music_test', opt.video2_ins, opt.video2_name)
	frame_name = os.listdir(clip_2_path)[0][:6]
	frame_name = float(frame_name)
	clip_det_bbs2 = None
	sign = False
	for n in range(detectionResult2.shape[0]):
		index = detectionResult2[n][0]
		index = float(index)
		if index == frame_name:
			if not sign:
				clip_det_bbs2 = np.expand_dims(detectionResult2[n,:],axis=0)
				sign = True
			else:
				clip_det_bbs2 = np.concatenate((clip_det_bbs2,np.expand_dims(detectionResult2[n,:],axis=0)),axis=0)
	# clip_det_bbs2 = sample_object_detections(detectionResult2)
	# clip_det_bbs3 = sample_object_detections(detectionResult3)

	avged_sep_audio1 = np.zeros((audio_length))
	avged_refine_audio1 = np.zeros((audio_length))
	avged_sep_audio2 = np.zeros((audio_length))
	avged_refine_audio2 = np.zeros((audio_length))
	avged_sep_audio3 = np.zeros((audio_length))
	avged_refine_audio3 = np.zeros((audio_length))

	for i in range(opt.num_of_object_detections_to_use):
		# 第二个的筛选

		# det_box2 = clip_det_bbs2[np.argmax(clip_det_bbs2[:, 2]), :]
		# clip_det_bbs2[np.argmax(clip_det_bbs2[:, 2]),2] = 0
		#
		# det_box3 = clip_det_bbs2[np.argmax(clip_det_bbs2[:, 2]), :]
		det_box2 = clip_det_bbs2[0,:]
		det_box3 = clip_det_bbs2[1,:]

		frame_path2 = os.path.join(opt.data_path_duet, 'duet_extract', opt.video2_ins, opt.video2_name,
									   "%06d.png" % det_box2[0])
		frame_2 = Image.open(frame_path2).convert('RGB')
		# frame = Image.open('/data/mashuo/work/study/refine-separation/dataset/music_test/xylophone-acoustic_guitar/0EMNATwzLA4_25/human.png').convert('RGB')
		detection2 = frame_2.crop((det_box2[-4],det_box2[-3],det_box2[-2],det_box2[-1]))
		# detection2 = frame

		detection3 = frame_2.crop((det_box3[-4],det_box3[-3],det_box3[-2],det_box3[-1]))


		#perform separation over the whole audio using a sliding window approach
		overlap_count = np.zeros((audio_length))
		sep_audio1 = np.zeros((audio_length))
		sep_audio2 = np.zeros((audio_length))
		sep_audio3 = np.zeros((audio_length))
		refine_sep1 = np.zeros((audio_length))
		refine_sep2 = np.zeros((audio_length))
		refine_sep3 = np.zeros((audio_length))

		sliding_window_start = 0
		data = {}
		samples_per_window = opt.audio_window
		while sliding_window_start + samples_per_window < audio_length:
			objects_visuals = []
			objects_labels = []
			objects_audio_mag = []
			objects_audio_phase = []
			objects_vids = []
			objects_real_audio_mag = []
			objects_audio_mix_mag = []
			objects_audio_mix_phase = []
			objects_visuals_256 = []

			sliding_window_end = sliding_window_start + samples_per_window
			audio_segment = audio_mix[sliding_window_start:sliding_window_end]
			audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop)


			''' 第二份音乐的信息'''
			objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
			objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
			objects_visuals.append(vision_transform(detection2).unsqueeze(0))
			objects_labels.append(torch.FloatTensor(np.ones((1, 1))))
			objects_vids.append(torch.FloatTensor(np.ones((1, 1))))

			''' 第3份音乐的信息'''
			objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
			objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
			objects_visuals.append(vision_transform(detection3).unsqueeze(0))
			objects_labels.append(torch.FloatTensor(np.ones((1, 1))))
			objects_vids.append(torch.FloatTensor(np.ones((1, 1))))

			data['audio_mix_mags'] = torch.FloatTensor(np.vstack(objects_audio_mix_mag)).cuda()
			data['audio_mags'] = data['audio_mix_mags']
			data['audio_mix_phases'] = torch.FloatTensor(np.vstack(objects_audio_mix_phase)).cuda()
			data['visuals'] = torch.FloatTensor(np.vstack(objects_visuals)).cuda()
			data['labels'] = torch.FloatTensor(np.vstack(objects_labels)).cuda()
			data['vids'] = torch.FloatTensor(np.vstack(objects_vids)).cuda()

			outputs = model.forward(data)

			reconstructed_signal, refine_signal = get_separated_audio(outputs, data, opt)

			sep_audio2[sliding_window_start:sliding_window_end] = sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal[0]
			refine_sep2[sliding_window_start:sliding_window_end] = refine_sep2[sliding_window_start:sliding_window_end] + refine_signal[0]

			sep_audio3[sliding_window_start:sliding_window_end] = sep_audio3[sliding_window_start:sliding_window_end] + reconstructed_signal[1]
			refine_sep3[sliding_window_start:sliding_window_end] = refine_sep3[sliding_window_start:sliding_window_end] + refine_signal[1]
			#update overlap count
			overlap_count[sliding_window_start:sliding_window_end] = overlap_count[sliding_window_start:sliding_window_end] + 1
			sliding_window_start = sliding_window_start + int(opt.hop_size * opt.audio_sampling_rate)

		# deal with the last segment
		audio_segment = audio_mix[-samples_per_window:]
		audio_mix_mags, audio_mix_phases = generate_spectrogram_magphase(audio_segment, opt.stft_frame, opt.stft_hop)

		objects_visuals = []
		objects_labels = []
		objects_audio_mag = []
		objects_audio_phase = []
		objects_vids = []
		objects_real_audio_mag = []
		objects_audio_mix_mag = []
		objects_audio_mix_phase = []
		objects_visuals_256 = []


		''' 第二份音乐的信息'''
		objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
		objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
		objects_visuals.append(vision_transform(detection2).unsqueeze(0))
		objects_labels.append(torch.FloatTensor(np.ones((1, 1))))
		objects_vids.append(torch.FloatTensor(np.ones((1, 1))))

		''' 第3份音乐的信息'''
		objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
		objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
		objects_visuals.append(vision_transform(detection3).unsqueeze(0))
		objects_labels.append(torch.FloatTensor(np.ones((1, 1))))
		objects_vids.append(torch.FloatTensor(np.ones((1, 1))))

		data['audio_mix_mags'] = torch.FloatTensor(np.vstack(objects_audio_mix_mag)).cuda()
		data['audio_mags'] = data['audio_mix_mags']
		data['audio_mix_phases'] = torch.FloatTensor(np.vstack(objects_audio_mix_phase)).cuda()
		data['visuals'] = torch.FloatTensor(np.vstack(objects_visuals)).cuda()
		data['labels'] = torch.FloatTensor(np.vstack(objects_labels)).cuda()
		data['vids'] = torch.FloatTensor(np.vstack(objects_vids)).cuda()

		outputs = model.forward(data)

		reconstructed_signal, refine_signal = get_separated_audio(outputs, data, opt)

		sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal[0]
		refine_sep2[-samples_per_window:] = refine_sep2[-samples_per_window:] + refine_signal[0]

		sep_audio3[-samples_per_window:] = sep_audio3[-samples_per_window:] + reconstructed_signal[1]
		refine_sep3[-samples_per_window:] = refine_sep3[-samples_per_window:] + refine_signal[1]

		#update overlap count
		overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

		#divide the aggregated predicted audio by the overlap count
		avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count) * 2)
		avged_refine_audio2 = avged_refine_audio2 + clip_audio(np.divide(refine_sep2, overlap_count) * 2)
		avged_sep_audio3 = avged_sep_audio3 + clip_audio(np.divide(sep_audio3, overlap_count) * 2)
		avged_refine_audio3 = avged_refine_audio3 + clip_audio(np.divide(refine_sep3, overlap_count) * 2)


	separation2 = avged_sep_audio2 / opt.num_of_object_detections_to_use
	separation3 = avged_sep_audio3 / opt.num_of_object_detections_to_use
	refine_spearation2 = avged_refine_audio2 / opt.num_of_object_detections_to_use
	refine_spearation3 = avged_refine_audio3 / opt.num_of_object_detections_to_use

	#output original and separated audios
	output_dir = os.path.join(output_dir, opt.video2_name +'**')
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if save_files:
		librosa.output.write_wav(os.path.join(output_dir, 'audio_duet.wav'), audio2, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio_mixed.wav'), audio_mix, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio2_separated.wav'), separation2, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio3_separated.wav'), separation3, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio2_refine_separated.wav'), refine_spearation2, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio3_refine_separated.wav'), refine_spearation3, opt.audio_sampling_rate)

	c_reference_sources = np.expand_dims(audio2, axis=0)
	c_estimated_sources = np.expand_dims((separation2+separation3), axis=0)

	c_sdr, c_sir, c_sar = getSeparationMetrics(c_reference_sources, c_estimated_sources)

	r_reference_sources = np.expand_dims(audio2, axis=0)
	r_estimated_sources = np.expand_dims((refine_spearation2+refine_spearation3), axis=0)

	r_sdr, r_sir, r_sar = getSeparationMetrics(r_reference_sources, r_estimated_sources)

	#save the two detections
	if save_files:
		frame_2.save(os.path.join(output_dir, 'frame_2.png'))
		detection2.save(os.path.join(output_dir, 'det2.png'))
		detection3.save(os.path.join(output_dir, 'det3.png'))
	#save the spectrograms & masks
	if opt.visualize_spectrogram:
		import matplotlib.pyplot as plt
		plt.switch_backend('agg')
		plt.ioff()
		audio2_mag = generate_spectrogram_magphase(audio2, opt.stft_frame, opt.stft_hop, with_phase=False)

		audio_mix_mag = generate_spectrogram_magphase(audio_mix, opt.stft_frame, opt.stft_hop, with_phase=False)
		separation2_mag = generate_spectrogram_magphase(separation2, opt.stft_frame, opt.stft_hop, with_phase=False)
		separation3_mag = generate_spectrogram_magphase(separation3, opt.stft_frame, opt.stft_hop, with_phase=False)
		refine_sep2_mag = generate_spectrogram_magphase(refine_spearation2, opt.stft_frame, opt.stft_hop, with_phase=False)
		refine_sep3_mag = generate_spectrogram_magphase(refine_spearation3, opt.stft_frame, opt.stft_hop, with_phase=False)
		# ref_2_3_mag = generate_spectrogram_magphase(refine_spearation1+refine_spearation2,  opt.stft_frame, opt.stft_hop, with_phase=False)
		utils.visualizeSpectrogram(audio2_mag[0,:,:], os.path.join(output_dir, 'audio2_spec.png'))
		utils.visualizeSpectrogram(audio_mix_mag[0,:,:], os.path.join(output_dir, 'audio_mixed_spec.png'))
		utils.visualizeSpectrogram(separation2_mag[0,:,:], os.path.join(output_dir, 'separation2_spec.png'))
		utils.visualizeSpectrogram(separation3_mag[0,:,:], os.path.join(output_dir, 'separation3_spec.png'))
		utils.visualizeSpectrogram(separation2_mag[0,:,:]+separation3_mag[0,:,:], os.path.join(output_dir, 'separation2+3_spec.png'))
		utils.visualizeSpectrogram(refine_sep2_mag[0, :, :], os.path.join(output_dir, 'refine2_spec.png'))
		utils.visualizeSpectrogram(refine_sep3_mag[0, :, :], os.path.join(output_dir, 'refine3_spec.png'))
		utils.visualizeSpectrogram(refine_sep2_mag[0, :, :]+refine_sep3_mag[0,:,:], os.path.join(output_dir, 'ref_2+3_spec.png'))
	return c_sdr, c_sir, c_sar, r_sdr, r_sir, r_sar

def get_vid_name(npy_path):
    #first 11 chars are the video id
    return os.path.basename(npy_path)[0:11]

def get_ins_name(npy_path):
    return os.path.basename(os.path.dirname(npy_path))

def get_clip_name(npy_path):
    return os.path.basename(npy_path)[0:-4]


if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "0"

	opt = TestOptions().parse()
	opt.device = torch.device("cuda")
	opt.mode = 'test'

	opt.data_path = opt.data_path.replace('/home/', opt.hri_change)
	opt.data_path_duet = opt.data_path_duet.replace('/home/', opt.hri_change)
	opt.hdf5_path = opt.hdf5_path.replace('/home/', opt.hri_change)

	# h5f_path = os.path.join(opt.hdf5_path, 'test.h5')
	# h5f = h5py.File(h5f_path, 'r')
	# detections = h5f['detection'][:]
	# solo_samples_list = []
	#
	# for detection in detections:
	# 	detection = detection.decode()
	# 	solo_samples_list.append(detection)

	duet_h5f_path = os.path.join(opt.hdf5_path, 'duet_test.h5')
	duet_h5f = h5py.File(duet_h5f_path, 'r')
	duet_detections = duet_h5f['detection'][:]
	duet_samples_list = []

	for d_detection in duet_detections:
		detection = d_detection.decode()
		duet_samples_list.append(detection)

	# solo_instruments = {'accordion':[],'acoustic_guitar':[],'cello':[],
	# 			   'clarinet':[],'erhu':[],'flute':[],'saxophone':[],
	# 			   'trumpet':[],'tuba':[],'violin':[],'xylophone':[]}
	# for sample in solo_samples_list:
	# 	ins = get_ins_name(sample)
	# 	solo_instruments[ins].append(get_clip_name(sample))

	duet_instruments = {'acoustic_guitar-violin': [], 'cello-acoustic_guitar': [], 'clarinet-acoustic_guitar': [],
						'flute-trumpet': [], 'flute-violin': [], 'saxophone-acoustic_guitar': [], 'trumpet-tuba': [],
						'xylophone-acoustic_guitar': [], 'xylophone-flute': []}

	category = 'trumpet-tuba'
	for sample in duet_samples_list:
		ins = get_ins_name(sample)
		if ins == category:
			duet_instruments[ins].append(get_clip_name(sample))



	# Network Builders
	builder = ModelBuilder()
	net_visual = builder.build_visual(
		pool_type=opt.visual_pool,
		weights=opt.weights_visual)
	net_unet = builder.build_unet(
		unet_num_layers=opt.unet_num_layers,
		ngf=opt.unet_ngf,
		input_nc=opt.unet_input_nc,
		output_nc=opt.unet_output_nc,
		weights=opt.weights_unet)
	# if opt.with_additional_scene_image:
	# 	opt.number_of_classes = opt.number_of_classes + 1
	net_classifier = builder.build_classifier(
		pool_type=opt.classifier_pool,
		num_of_classes=opt.number_of_classes + 1,
		input_channel=opt.unet_output_nc,
		weights=opt.weights_classifier)
	net_refine = builder.build_refine(
		opt=opt,
		ngf=opt.unet_ngf,
		input_nc=2,
		output_nc=opt.unet_output_nc,
		weights=opt.weights_refine)
	net_audio_extractor = builder.build_down_channels(weights=opt.weights_down_channels)

	nets = (net_visual, net_unet, net_classifier, net_refine, net_audio_extractor)

	# checkpoint_weights = './checkpoints/sup10_music21_train_2mix/checkpoint_latest.pth'
	checkpoint_weights = './checkpoints/sup1_iteration_factor_arithmetic_down_2mix/checkpoint_latest.pth'

	ckpt = torch.load(checkpoint_weights)
	net_visual.load_state_dict(ckpt['state_dict_visual'])
	net_unet.load_state_dict(ckpt['state_dict_unet'])
	net_refine.load_state_dict(ckpt['state_dict_refine'])
	net_classifier.load_state_dict(ckpt['state_dict_classifier'])
	net_audio_extractor.load_state_dict(ckpt['state_dict_down'])

	opt.batch_size = 1

	c_sdr_all = []
	c_sir_all = []
	c_sar_all = []
	r_sdr_all = []
	r_sir_all = []
	r_sar_all = []

	test_name = 'test_duet_on_by_one'
	# output dir
	output_dir = os.path.join(opt.output_dir_root, test_name, category)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)


	fp = open(os.path.join(output_dir, 'align_{}.csv'.format(test_name)), 'w')
	writer = csv.writer(fp)
	writer.writerow(['No.', 'duet_ins','first', 'another', 'c_sdr', 'c_sar', 'c_sir', 'r_sdr', 'r_sar', 'r_sir'])

	duet_ins_root = '/data/mashuo/work/study/refine-separation/dataset/music_test'

	# solo_inses = list(solo_instruments.keys())
	duet_inses = list(duet_instruments.keys())
	# test_pairs = list(combinations(ins, 3))  # 这里的3实际为混合的音频个数

	# total_tests = 0
	# for solo_ins in solo_inses:
	# for duet_ins in duet_inses:
	# duet_1,duet_2 = duet_ins.split('-')
	# if duet_1 == solo_ins or duet_2 == solo_ins:
	# 	print('class cover: {}/{}'.format(solo_ins,duet_ins))
	# 	continue
	# opt.video1_ins = solo_ins
	opt.video2_ins = category
	# opt.video1_name = random.choice(solo_instruments[opt.video1_ins])
	# opt.video2_name = os.listdir(os.path.join(duet_ins_root,category))[0]
	opt.video2_name = 'Zqg4DwXmYBI_2'
	# opt.video1_name = random.choice(solo_instruments[opt.video1_ins])
	# opt.video2_name = random.choice(duet_instruments[opt.video2_ins])

	c_sdr, c_sir, c_sar, r_sdr, r_sir, r_sar= test_sepration(opt, nets, output_dir, save_files=True)
	# total_tests += 1
	c_sdr_all.append(c_sdr)
	c_sir_all.append(c_sir)
	c_sar_all.append(c_sar)
	r_sdr_all.append(r_sdr)
	r_sir_all.append(r_sir)
	r_sar_all.append(r_sar)
	print("|{}|:sep {}, c_sdr:{}, c_sir:{}, c_sar:{}, r_sdr:{}, r_sir:{}, r_sar:{}\n".format(
		1,
		opt.video2_name,
		c_sdr, c_sir, c_sar,
		r_sdr, r_sir, r_sar))
	writer.writerow(
		[1, opt.video2_ins,opt.video1_name, opt.video2_name,
		 c_sdr, c_sir, c_sar,
		 r_sdr, r_sir, r_sar])
	# print('total are:{},{},{},{},{},{}'.format(np.mean(c_sdr_all), np.mean(c_sir_all), np.mean(c_sar_all),
	# 										   np.mean(r_sdr_all), np.mean(r_sir_all), np.mean(r_sar_all)))
	# # writer.writerow(['average result on test dataset:{}/{}'.format(total_tests, len(samples_list))])
	# writer.writerow([np.mean(c_sdr_all), np.mean(c_sir_all), np.mean(c_sar_all), np.mean(r_sdr_all), np.mean(r_sir_all),
	# 				 np.mean(r_sar_all)])

	fp.close()
