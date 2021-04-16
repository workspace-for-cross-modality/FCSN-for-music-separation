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
from data.audioVisual_dataset import sample_object_detections
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
	refine_masks = [None for i in range(opt.refine_iteration)]

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

	for n in range(opt.refine_iteration):
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
	audio1_path = os.path.join(opt.data_path, 'solo_audio_resample', opt.video1_ins, opt.video1_name + '.wav')
	audio1, _ = librosa.load(audio1_path, sr=opt.audio_sampling_rate)
	audio2_path = os.path.join(opt.data_path, 'solo_audio_resample', opt.video2_ins, opt.video2_name + '.wav')
	audio2, _ = librosa.load(audio2_path, sr=opt.audio_sampling_rate)



	#make sure the two audios are of the same length and then mix them
	audio_length = min(len(audio1), len(audio2))
	audio1 = clip_audio(audio1[:audio_length])
	audio2 = clip_audio(audio2[:audio_length])
	audio_mix = (audio1 + audio2) / 2.0

	#define the transformation to perform on visual frames
	vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
	vision_transform_list_for_unet = [transforms.Resize((256, 256)), transforms.ToTensor()]
	if opt.subtract_mean:
		vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
		vision_transform_list_for_unet.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))

	vision_transform = transforms.Compose(vision_transform_list)
	vision_transform_for_unet = transforms.Compose(vision_transform_list_for_unet)

	#load the object regions of the highest confidence score for both videos
	detectionResult1 = np.load(os.path.join(opt.data_path, 'solo_detect', opt.video1_ins, opt.video1_name + '.npy'))
	detectionResult2 = np.load(os.path.join(opt.data_path, 'solo_detect', opt.video2_ins, opt.video2_name + '.npy'))

	clip_det_bbs1 = sample_object_detections(detectionResult1)
	clip_det_bbs2 = sample_object_detections(detectionResult2)


	avged_sep_audio1 = np.zeros((audio_length))
	avged_refine_audio1 = np.zeros((audio_length))
	avged_sep_audio2 = np.zeros((audio_length))
	avged_refine_audio2 = np.zeros((audio_length))


	for i in range(opt.num_of_object_detections_to_use):
		# 第一个的筛选
		if clip_det_bbs1.shape[0] == 1:
			frame_path1 = os.path.join(opt.data_path, 'solo_extract', opt.video1_ins, opt.video1_name, "%06d.png" % clip_det_bbs1[0,0])
			detection_bbs_filter_1 = clip_det_bbs1[0]
		elif clip_det_bbs1.shape[0] >= 2:
			hand_npy = os.path.join(opt.data_path,'solo_detect_hand',opt.video1_ins,opt.video1_name + '_hand.npy')
			if os.path.exists(hand_npy):
				hand_bbs = np.load(hand_npy)
			else:
				hand_bbs = np.array([])

			if hand_bbs.shape[0] == 0:
				hand_bb = np.array([])
				sign = False
				print("this npy file {} donot have detected hands".format(os.path.basename(hand_npy)))
			elif hand_bbs.shape[0] == 1:
				hand_bb = hand_bbs
				sign = True
			elif hand_bbs.shape[0] >= 2:  # 在检测到的乐器数不止一个的情况下，如果检测到两只手以上，则取计算结果中概率最大的前两个
				the_max = np.argmax(hand_bbs[:, 1])
				hand_bb1 = hand_bbs[the_max, :]  # 取一个概率最大的
				hand_bb1 = hand_bb1[np.newaxis, :]
				hand_bbs[the_max, 1] = 0  # 取出后置为0
				the_second_max = np.argmax(hand_bbs[:, 1])  # 取一个次大的。
				hand_bb2 = hand_bbs[the_second_max, :]
				hand_bb2 = hand_bb2[np.newaxis, :]
				hand_bb = np.concatenate((hand_bb1, hand_bb2), axis=0)
				sign = True
			detection_bbs_filter_1 = filter_det_bbs(hand_bb, sign, clip_det_bbs1)
			frame_path1 = os.path.join(opt.data_path, 'solo_extract', opt.video1_ins, opt.video1_name,
									   "%06d.png" % detection_bbs_filter_1[0])

		detection1 = Image.open(frame_path1).convert('RGB').crop((detection_bbs_filter_1[-4],
																  detection_bbs_filter_1[-3],
																  detection_bbs_filter_1[-2],
																  detection_bbs_filter_1[-1]))
		# 第二个的筛选
		if clip_det_bbs2.shape[0] == 1:
			frame_path2 = os.path.join(opt.data_path, 'solo_extract', opt.video2_ins, opt.video2_name, "%06d.png" % clip_det_bbs2[0,0])
			detection_bbs_filter_2 = clip_det_bbs2[0]
		elif clip_det_bbs2.shape[0] >= 2:
			hand_npy = os.path.join(opt.data_path,'solo_detect_hand',opt.video2_ins,opt.video2_name + '_hand.npy')
			if os.path.exists(hand_npy):
				hand_bbs = np.load(hand_npy)
			else:
				hand_bbs = np.array([])

			if hand_bbs.shape[0] == 0:
				hand_bb = np.array([])
				sign = False
				print("this npy file {} donot have detected hands".format(os.path.basename(hand_npy)))
			elif hand_bbs.shape[0] == 1:
				hand_bb = hand_bbs
				sign = True
			elif hand_bbs.shape[0] >= 2:  # 在检测到的乐器数不止一个的情况下，如果检测到两只手以上，则取计算结果中概率最大的前两个
				the_max = np.argmax(hand_bbs[:, 1])
				hand_bb1 = hand_bbs[the_max, :]  # 取一个概率最大的
				hand_bb1 = hand_bb1[np.newaxis, :]
				hand_bbs[the_max, 1] = 0  # 取出后置为0
				the_second_max = np.argmax(hand_bbs[:, 1])  # 取一个次大的。
				hand_bb2 = hand_bbs[the_second_max, :]
				hand_bb2 = hand_bb2[np.newaxis, :]
				hand_bb = np.concatenate((hand_bb1, hand_bb2), axis=0)
				sign = True
			detection_bbs_filter_2 = filter_det_bbs(hand_bb, sign, clip_det_bbs2)
			frame_path2 = os.path.join(opt.data_path, 'solo_extract', opt.video2_ins, opt.video2_name,
									   "%06d.png" % clip_det_bbs2[0, 0])

		detection2 = Image.open(frame_path2).convert('RGB').crop((detection_bbs_filter_2[-4],
																  detection_bbs_filter_2[-3],
																  detection_bbs_filter_2[-2],
																  detection_bbs_filter_2[-1]))


		#perform separation over the whole audio using a sliding window approach
		overlap_count = np.zeros((audio_length))
		sep_audio1 = np.zeros((audio_length))
		sep_audio2 = np.zeros((audio_length))
		refine_sep1 = np.zeros((audio_length))
		refine_sep2 = np.zeros((audio_length))

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

			''' 第一份音乐的信息'''
			objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
			objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
			objects_visuals.append(vision_transform(detection1).unsqueeze(0))
			objects_visuals_256.append(vision_transform_for_unet(detection1).unsqueeze(0))
			objects_labels.append(torch.FloatTensor(np.ones((1,1))))
			objects_vids.append(torch.FloatTensor(np.ones((1,1))))

			''' 第二份音乐的信息'''
			objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
			objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
			objects_visuals.append(vision_transform(detection2).unsqueeze(0))
			objects_visuals_256.append(vision_transform_for_unet(detection2).unsqueeze(0))
			objects_labels.append(torch.FloatTensor(np.ones((1, 1))))
			objects_vids.append(torch.FloatTensor(np.ones((1, 1))))


			data['audio_mix_mags'] = torch.FloatTensor(np.vstack(objects_audio_mix_mag)).cuda()
			data['audio_mags'] = data['audio_mix_mags']
			data['audio_mix_phases'] = torch.FloatTensor(np.vstack(objects_audio_mix_phase)).cuda()
			data['visuals'] = torch.FloatTensor(np.vstack(objects_visuals)).cuda()
			data['visuals_256'] = torch.FloatTensor(np.vstack(objects_visuals_256)).cuda()
			data['labels'] = torch.FloatTensor(np.vstack(objects_labels)).cuda()
			data['vids'] = torch.FloatTensor(np.vstack(objects_vids)).cuda()

			outputs = model.forward(data)

			reconstructed_signal, refine_signal = get_separated_audio(outputs, data, opt)

			sep_audio1[sliding_window_start:sliding_window_end] = sep_audio1[sliding_window_start:sliding_window_end] + reconstructed_signal[0]
			refine_sep1[sliding_window_start:sliding_window_end] = refine_sep1[sliding_window_start:sliding_window_end] + refine_signal[0]

			sep_audio2[sliding_window_start:sliding_window_end] = sep_audio2[sliding_window_start:sliding_window_end] + reconstructed_signal[1]
			refine_sep2[sliding_window_start:sliding_window_end] = refine_sep2[sliding_window_start:sliding_window_end] + refine_signal[1]

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

		''' 第一份音乐的信息，应该有两份'''
		objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
		objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
		objects_visuals.append(vision_transform(detection1).unsqueeze(0))
		objects_visuals_256.append(vision_transform_for_unet(detection1).unsqueeze(0))
		objects_labels.append(torch.FloatTensor(np.ones((1, 1))))
		objects_vids.append(torch.FloatTensor(np.ones((1, 1))))

		''' 第二份音乐的信息'''
		objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mags).unsqueeze(0))
		objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phases).unsqueeze(0))
		objects_visuals_256.append(vision_transform_for_unet(detection2).unsqueeze(0))
		objects_visuals.append(vision_transform(detection2).unsqueeze(0))
		objects_labels.append(torch.FloatTensor(np.ones((1, 1))))
		objects_vids.append(torch.FloatTensor(np.ones((1, 1))))


		data['audio_mix_mags'] = torch.FloatTensor(np.vstack(objects_audio_mix_mag)).cuda()
		data['audio_mags'] = data['audio_mix_mags']
		data['audio_mix_phases'] = torch.FloatTensor(np.vstack(objects_audio_mix_phase)).cuda()
		data['visuals'] = torch.FloatTensor(np.vstack(objects_visuals)).cuda()
		data['labels'] = torch.FloatTensor(np.vstack(objects_labels)).cuda()
		data['vids'] = torch.FloatTensor(np.vstack(objects_vids)).cuda()
		data['visuals_256'] = torch.FloatTensor(np.vstack(objects_visuals_256)).cuda()

		outputs = model.forward(data)

		reconstructed_signal, refine_signal = get_separated_audio(outputs, data, opt)
		sep_audio1[-samples_per_window:] = sep_audio1[-samples_per_window:] + reconstructed_signal[0]
		refine_sep1[-samples_per_window:] = refine_sep1[-samples_per_window:] + refine_signal[0]

		sep_audio2[-samples_per_window:] = sep_audio2[-samples_per_window:] + reconstructed_signal[1]
		refine_sep2[-samples_per_window:] = refine_sep2[-samples_per_window:] + refine_signal[1]

		#update overlap count
		overlap_count[-samples_per_window:] = overlap_count[-samples_per_window:] + 1

		#divide the aggregated predicted audio by the overlap count
		avged_sep_audio1 = avged_sep_audio1 + clip_audio(np.divide(sep_audio1, overlap_count) * 2)
		avged_refine_audio1 = avged_refine_audio1 + clip_audio(np.divide(refine_sep1, overlap_count)*2)
		avged_sep_audio2 = avged_sep_audio2 + clip_audio(np.divide(sep_audio2, overlap_count) * 2)
		avged_refine_audio2 = avged_refine_audio2 + clip_audio(np.divide(refine_sep2, overlap_count) * 2)


	separation1 = avged_sep_audio1 / opt.num_of_object_detections_to_use
	separation2 = avged_sep_audio2 / opt.num_of_object_detections_to_use

	refine_spearation1 = avged_refine_audio1 / opt.num_of_object_detections_to_use
	refine_spearation2 = avged_refine_audio2 / opt.num_of_object_detections_to_use


	#output original and separated audios
	output_dir = os.path.join(output_dir, opt.video1_name +'$_VS_$' + opt.video2_name)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)

	if save_files:
		librosa.output.write_wav(os.path.join(output_dir, 'audio1.wav'), audio1, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio2.wav'), audio2, opt.audio_sampling_rate)

		librosa.output.write_wav(os.path.join(output_dir, 'audio_mixed.wav'), audio_mix, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio1_separated.wav'), separation1, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio2_separated.wav'), separation2, opt.audio_sampling_rate)

		librosa.output.write_wav(os.path.join(output_dir, 'audio1_refine_separated.wav'), refine_spearation1, opt.audio_sampling_rate)
		librosa.output.write_wav(os.path.join(output_dir, 'audio2_refine_separated.wav'), refine_spearation2, opt.audio_sampling_rate)


	c_reference_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
	c_estimated_sources = np.concatenate((np.expand_dims(separation1, axis=0), np.expand_dims(separation2, axis=0)), axis=0)

	c_sdr, c_sir, c_sar = getSeparationMetrics(c_reference_sources, c_estimated_sources)

	r_reference_sources = np.concatenate((np.expand_dims(audio1, axis=0), np.expand_dims(audio2, axis=0)), axis=0)
	r_estimated_sources = np.concatenate((np.expand_dims(refine_spearation1, axis=0),
										  np.expand_dims(refine_spearation2, axis=0)), axis=0)

	r_sdr, r_sir, r_sar = getSeparationMetrics(r_reference_sources, r_estimated_sources)

	#save the two detections
	if save_files:
		detection1.save(os.path.join(output_dir, 'audio1.png'))
		detection2.save(os.path.join(output_dir, 'audio2.png'))
	#save the spectrograms & masks
	if opt.visualize_spectrogram:
		import matplotlib.pyplot as plt
		plt.switch_backend('agg')
		plt.ioff()
		audio1_mag = generate_spectrogram_magphase(audio1, opt.stft_frame, opt.stft_hop, with_phase=False)
		audio2_mag = generate_spectrogram_magphase(audio2, opt.stft_frame, opt.stft_hop, with_phase=False)

		audio_mix_mag = generate_spectrogram_magphase(audio_mix, opt.stft_frame, opt.stft_hop, with_phase=False)
		separation1_mag = generate_spectrogram_magphase(separation1, opt.stft_frame, opt.stft_hop, with_phase=False)
		separation2_mag = generate_spectrogram_magphase(separation2, opt.stft_frame, opt.stft_hop, with_phase=False)

		refine_sep1_mag = generate_spectrogram_magphase(refine_spearation1, opt.stft_frame, opt.stft_hop, with_phase=False)
		refine_sep2_mag = generate_spectrogram_magphase(refine_spearation2, opt.stft_frame, opt.stft_hop, with_phase=False)

		utils.visualizeSpectrogram(audio1_mag[0,:,:], os.path.join(output_dir, 'audio1_spec.png'))
		utils.visualizeSpectrogram(audio2_mag[0,:,:], os.path.join(output_dir, 'audio2_spec.png'))

		utils.visualizeSpectrogram(audio_mix_mag[0,:,:], os.path.join(output_dir, 'audio_mixed_spec.png'))
		utils.visualizeSpectrogram(separation1_mag[0,:,:], os.path.join(output_dir, 'separation1_spec.png'))
		utils.visualizeSpectrogram(separation2_mag[0,:,:], os.path.join(output_dir, 'separation2_spec.png'))

		utils.visualizeSpectrogram(refine_sep1_mag[0, :, :], os.path.join(output_dir, 'refine1_spec.png'))
		utils.visualizeSpectrogram(refine_sep2_mag[0, :, :], os.path.join(output_dir, 'refine2_spec.png'))

	return c_sdr, c_sir, c_sar, r_sdr, r_sir, r_sar

def get_vid_name(npy_path):
    #first 11 chars are the video id
    return os.path.basename(npy_path)[0:11]

def get_ins_name(npy_path):
    return os.path.basename(os.path.dirname(npy_path))

def get_clip_name(npy_path):
    return os.path.basename(npy_path)[0:-4]


if __name__ == '__main__':

	os.environ["CUDA_VISIBLE_DEVICES"] = "4"

	opt = TestOptions().parse()
	opt.device = torch.device("cuda")
	opt.mode = 'test'

	opt.data_path = opt.data_path.replace('/home/', opt.hri_change)
	opt.data_path_duet = opt.data_path_duet.replace('/home/', opt.hri_change)
	opt.hdf5_path = opt.hdf5_path.replace('/home/', opt.hri_change)

	h5f_path = os.path.join(opt.hdf5_path,  'test.h5')
	h5f = h5py.File(h5f_path, 'r')
	detections = h5f['detection'][:]
	samples_list = []

	for detection in detections:
		detection = detection.decode()
		samples_list.append(detection)
	# instruments = {'accordion':[],'acoustic_guitar':[],'cello':[],
	# 			   'clarinet':[],'erhu':[],'flute':[],'saxophone':[],
	# 			   'trumpet':[],'tuba':[],'violin':[],'xylophone':[]}

	instruments = {'accordion': [], 'acoustic_guitar': [], 'bagpipe': [], 'banjo': [],
				   'bassoon': [], 'cello': [], 'clarinet': [], 'congas': [], 'drum': [],
				   'electric_bass': [], 'erhu': [], 'flute': [], 'guzheng': [],
				   'piano': [], 'pipa': [], 'saxophone': [], 'trumpet': [], 'tuba': [],
				   'ukulele': [], 'violin': [], 'xylophone': []}
	for sample in samples_list:
		ins = get_ins_name(sample)
		instruments[ins].append(get_clip_name(sample))

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

	checkpoint_weights = './checkpoints/sup10_music21_train_2mix/checkpoint_latest.pth'

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

	test_name = opt.test_name
	# output dir
	output_dir = os.path.join(opt.output_dir_root, test_name)
	if not os.path.exists(output_dir):
		os.makedirs(output_dir)
	fp = open(os.path.join(output_dir, 'align_{}.csv'.format(test_name)),'w')
	writer = csv.writer(fp)
	writer.writerow(['No.','first', 'another', 'c_sdr', 'c_sar', 'c_sir','r_sdr', 'r_sar', 'r_sir'])

	ins = list(instruments.keys())
	test_pairs = list(combinations(ins,2))  # 这里的2实际为混合的音频个数



	total_tests = 0

	for iters in range(10):
		for pair in test_pairs:
			opt.video1_ins = pair[0]
			opt.video2_ins = pair[1]

			opt.video1_name = random.choice(instruments[opt.video1_ins])
			opt.video2_name = random.choice(instruments[opt.video2_ins])


			c_sdr, c_sir, c_sar, r_sdr, r_sir, r_sar = test_sepration(opt, nets, output_dir, save_files=True)
			total_tests += 1
			c_sdr_all.append(c_sdr)
			c_sir_all.append(c_sir)
			c_sar_all.append(c_sar)
			r_sdr_all.append(r_sdr)
			r_sir_all.append(r_sir)
			r_sar_all.append(r_sar)
			print("|{}|:sep {} with {}, c_sdr:{}, c_sir:{}, c_sar:{}, r_sdr:{}, r_sir:{}, r_sar:{}\n".format(total_tests,
																										  opt.video1_name,
																										  opt.video2_name,
																										  c_sdr,c_sir, c_sar,
																										  r_sdr, r_sir,r_sar))
			writer.writerow([total_tests, opt.video1_ins, opt.video2_ins, opt.video1_name,opt.video2_name, c_sdr, c_sir, c_sar, r_sdr, r_sir, r_sar])
	print('total are:{},{},{},{},{},{}'.format(np.mean(c_sdr_all), np.mean(c_sir_all), np.mean(c_sar_all), np.mean(r_sdr_all), np.mean(r_sir_all), np.mean(r_sar_all)))
	writer.writerow(['average result on test dataset:{}/{}'.format(total_tests, len(samples_list))])
	writer.writerow([np.mean(c_sdr_all), np.mean(c_sir_all), np.mean(c_sar_all), np.mean(r_sdr_all), np.mean(r_sir_all), np.mean(r_sar_all)])

	fp.close()
