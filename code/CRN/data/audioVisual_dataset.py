import os.path
import librosa
import h5py
import random
from random import randrange
import glob
from PIL import Image, ImageEnhance, ImageOps
import numpy as np
import torchvision.transforms as transforms
import torch
import math
import copy
import torch.utils.data as torchdata


def generate_spectrogram_magphase(audio, stft_frame, stft_hop, with_phase=True):
    spectro = librosa.core.stft(audio, hop_length=stft_hop, n_fft=stft_frame, center=True)
    spectro_mag, spectro_phase = librosa.core.magphase(spectro)
    spectro_mag = np.expand_dims(spectro_mag, axis=0)
    if with_phase:
        spectro_phase = np.expand_dims(np.angle(spectro_phase), axis=0)
        return spectro_mag, spectro_phase
    else:
        return spectro_mag

def augment_audio(audio):
    audio = audio * (random.random() + 0.5) # 0.5 - 1.5
    audio[audio > 1.] = 1.
    audio[audio < -1.] = -1.
    return audio

def sample_audio(audio, window):
    # repeat if audio is too short
    if audio.shape[0] < window:
        n = int(window / audio.shape[0]) + 1
        audio = np.tile(audio, n)
    audio_start = randrange(0, audio.shape[0] - window + 1)
    audio_sample = audio[audio_start:(audio_start+window)]
    return audio_sample

def augment_image(image):
	if(random.random() < 0.5):
		image = image.transpose(Image.FLIP_LEFT_RIGHT)
	enhancer = ImageEnhance.Brightness(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	enhancer = ImageEnhance.Color(image)
	image = enhancer.enhance(random.random()*0.6 + 0.7)
	return image

def get_vid_name(npy_path):
    #first 11 chars are the video id
    return os.path.basename(npy_path)[0:11]

def get_clip_name(npy_path):
    return os.path.basename(npy_path)[0:-4]

def get_frame_root(npy_path):
    a = os.path.dirname(os.path.dirname(npy_path))
    return os.path.join(os.path.dirname(a), 'solo_extract')

def get_ins_name(npy_path):
    return os.path.basename(os.path.dirname(npy_path))

def get_audio_root(npy_path):
    a = os.path.dirname(os.path.dirname(npy_path))
    return os.path.join(os.path.dirname(a), 'solo_audio_resample')


def get_hand_npy_path(path):
    root = os.path.dirname(os.path.dirname(os.path.dirname(path)))
    file_path = os.path.join(root, 'solo_detect_hand', get_ins_name(path), get_clip_name(path) + '_hand.npy')
    return file_path


def cal_centers(rects):
    center_x = (rects[0] + rects[2]) / 2
    center_y = (rects[1] + rects[3]) / 2
    return [center_x, center_y]


def cal_angle(v1, v2):
    dx1 = v1[2] - v1[0]
    dy1 = v1[3] - v1[1]
    dx2 = v2[2] - v2[0]
    dy2 = v2[3] - v2[1]
    angle1 = math.atan2(dy1, dx1)
    angle1 = math.degrees(angle1)
    # angle1 = int(angle1 * 180 / math.pi)
    # print(angle1)
    angle2 = math.atan2(dy2, dx2)
    angle2 = math.degrees(angle2)
    # angle2 = int(angle2 * 180 / math.pi)
    # print(angle2)
    if angle1 * angle2 >= 0:
        included_angle = abs(angle1 - angle2)
    else:
        included_angle = abs(angle1) + abs(angle2)
        if included_angle > 180:
            included_angle = 360 - included_angle
    return included_angle


def sample_object_detections(detection_bbs):
    class_index_clusters = {} #get the indexes of the detections for each class
    for i in range(detection_bbs.shape[0]):
        if class_index_clusters.__contains__(int(detection_bbs[i,1])):
            class_index_clusters[int(detection_bbs[i,1])].append(i)
        else:
            class_index_clusters[int(detection_bbs[i,1])] = [i]
    detection2return = np.array([])
    for cls in class_index_clusters.keys():
        sampledIndex = random.choice(class_index_clusters[cls])
        if detection2return.shape[0] == 0:
            detection2return = np.expand_dims(detection_bbs[sampledIndex,:], axis=0)
        else:
            # 每个类别中随机取一个，然后在原ndarray中取出对应行，保存到下面这个ndarray中
            detection2return = np.concatenate((detection2return, np.expand_dims(detection_bbs[sampledIndex,:], axis=0)), axis=0)
    return detection2return


def filter_det_bbs(hand_bb, sign, detection_bbs, alpha=100, beta=90):
    # todo: 这里的选择策略有问题，在复杂场景（多人演奏时），会造成误判
    # 如果没检测到手 , 返回置信度最高的乐器检测结果
    if not sign:
        ind = np.argmax(detection_bbs[:, 2])
        return detection_bbs[ind, :]

    if hand_bb.shape[0] == 1:   # hand_bb.any()表示其中任意一个是否有大于0的，如果有，则为True
        center_hand = [[float(hand_bb[0, 2]), float(hand_bb[0, 3])]]  # 一只手的中心点
        center_bbs = []
        for i in range(detection_bbs.shape[0]):
            center_bbs.append(cal_centers(detection_bbs[i, -4:]))
        res_cor = np.array(center_bbs) - np.array(center_hand)   # 各bbs坐标与hand中心坐标的差值，用于计算距离
        distances = np.array([math.hypot(res_cor[i][0], res_cor[i][1]) for i in range(len(res_cor))])
        # distances = distances[distances>alpha]
        distances = distances[:, np.newaxis]

        v1 = [hand_bb[0, 2], hand_bb[0, 3], hand_bb[0, 4], hand_bb[0, 5]]
        v1 = list(map(float,v1))
        angles = []
        for i in range(detection_bbs.shape[0]):
            v2 = [center_hand[0][0], center_hand[0][1], center_bbs[i][0], center_bbs[i][1]]
            angles.append(cal_angle(v1, v2))
        angles = np.array(angles)
        angles = angles[:, np.newaxis]
        distances_filtered = copy.deepcopy(distances)
        distances_filtered[angles[:, 0] > beta, 0] = float('inf')
        ind = np.argmin(distances_filtered[:, 0])
        if distances_filtered[ind,0] != float('inf'):
            return detection_bbs[ind]
        else:
            ind = np.argmin(distances[:,0])
            return detection_bbs[ind]

    elif hand_bb.shape[0] >=2:
        center_hand = [[float(hand_bb[0,2]),float(hand_bb[0,3])],[float(hand_bb[1,2]),float(hand_bb[1,3])]]  # 两只手的中心点
        center_bbs = []
        for i in range(detection_bbs.shape[0]):
            center_bbs.append(cal_centers(detection_bbs[i,-4:]))
        res_cor_1 = np.array(center_bbs) - np.array(center_hand[0])
        res_cor_2 = np.array(center_bbs) - np.array(center_hand[1])
        distances_1 = np.array([math.hypot(res_cor_1[i][0], res_cor_1[i][1]) for i in range(len(res_cor_1))])
        distances_1 = distances_1[:, np.newaxis]
        distances_2 = np.array([math.hypot(res_cor_2[i][0], res_cor_2[i][1]) for i in range(len(res_cor_2))])
        distances_2 = distances_2[:, np.newaxis]

        v1_1 = [hand_bb[0,2],hand_bb[0,3],hand_bb[0,4],hand_bb[0,5]]  # 第一只手的向量
        v1_1 = list(map(float,v1_1))
        v1_2 = [hand_bb[1,2],hand_bb[1,3],hand_bb[1,4],hand_bb[1,5]]  # 第二只手的向量
        v1_2 = list(map(float,v1_2))
        angles_1 = []
        angles_2 = []
        for i in range(detection_bbs.shape[0]):
            v2_1 = [center_hand[0][0],center_hand[0][1],center_bbs[i][0],center_bbs[i][1]]
            v2_1 = list(map(float,v2_1))
            v2_2 = [center_hand[1][0],center_hand[1][1],center_bbs[i][0],center_bbs[i][1]]
            v2_2 = list(map(float,v2_2))
            angles_1.append(cal_angle(v1_1,v2_1))
            angles_2.append(cal_angle(v1_2,v2_2))
        angles_1 = np.array(angles_1)
        angles_1 = angles_1[:, np.newaxis]
        angles_2 = np.array(angles_2)
        angles_2 = angles_2[:, np.newaxis]
        distances_1_filtered = copy.deepcopy(distances_1)
        distances_2_filtered = copy.deepcopy(distances_2)
        distances_1_filtered[angles_1[:,0]>beta, 0] = float('inf')
        distances_2_filtered[angles_2[:,0]>beta, 0] = float('inf')
        distances_sum = distances_1_filtered + distances_2_filtered
        min_dis_ind = np.argmin(distances_sum[:,0])
        if distances_sum[min_dis_ind,0] != float('inf'):
            return detection_bbs[min_dis_ind]
        else:
            distances_sum = distances_1 + distances_2
            min_dis_ind = np.argmin(distances_sum[:,0])
            return detection_bbs[min_dis_ind]




class AudioVisualMUSICDataset(torchdata.Dataset):
    def __init__(self,mode,opt):
        super(AudioVisualMUSICDataset, self).__init__()
        self.mode = mode
        self.opt = opt
        self.NUM_PER_MIX = opt.num_per_mix
        self.stft_frame = opt.stft_frame
        self.stft_hop = opt.stft_hop
        self.audio_window = opt.audio_window
        random.seed(opt.seed)

        #initialization
        self.detection_dic = {} #gather the clips for each video
        self.list_sample = []
        #load detection hdf5 file
        h5f_path = os.path.join(opt.hdf5_path, self.mode+'.h5')
        h5f = h5py.File(h5f_path, 'r')
        detections = h5f['detection'][:]
        for detection in detections:
            detection = detection.decode()

            # change to hri-3, the path is different
            detection = detection.replace('/home/',opt.hri_change)  # 切换二号三号机
            self.list_sample.append(detection)
            vidname = get_vid_name(detection) #get video id

            if self.detection_dic.__contains__(vidname):
                self.detection_dic[vidname].append(detection)
            else:
                self.detection_dic[vidname] = [detection]

        if self.mode == 'val':
            vision_transform_list = [transforms.Resize((224,224)), transforms.ToTensor()]
        elif opt.preserve_ratio:
            vision_transform_list = [transforms.Resize(256), transforms.RandomCrop(224), transforms.ToTensor()]
        else:
            vision_transform_list = [transforms.Resize((256, 256)), transforms.RandomCrop(224), transforms.ToTensor()]  # train use this
        if opt.subtract_mean:
            vision_transform_list.append(transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]))
        self.vision_transform = transforms.Compose(vision_transform_list)

    def __getitem__(self, index):
        N = self.NUM_PER_MIX

        audios = [None for n in range(self.NUM_PER_MIX)] #audios of mixed videos
        objects_visuals = []
        objects_labels = []
        objects_audio_mag = []
        objects_audio_phase = []
        objects_vids = []
        objects_audio_mix_mag = []
        objects_audio_mix_phase = []
        clip_paths = [None for n in range(N)]
        clip_audio_paths = [None for n in range(N)]
        clip_det_bbs = [None for n in range(N)]

        ''' 1st video'''
        clip_paths[0] = self.list_sample[index]
        clip_audio_paths[0] = os.path.join(get_audio_root(clip_paths[0]), get_ins_name(clip_paths[0]), get_clip_name(clip_paths[0]) + ".wav")
        category_lib = []
        category_lib.append(get_ins_name(clip_paths[0]))
        clip_det_bbs[0] = sample_object_detections(np.load(clip_paths[0]))

        for n in range(1, N):
            clip_paths[n] = random.choice(self.list_sample)
            category = get_ins_name(clip_paths[n])
            while category in category_lib:
                clip_paths[n] = random.choice(self.list_sample)
                category = get_ins_name(clip_paths[n])
            category_lib.append(category)
            clip_audio_paths[n] = os.path.join(get_audio_root(clip_paths[n]), get_ins_name(clip_paths[n]),get_clip_name(clip_paths[n]) + ".wav")
            clip_det_bbs[n] = sample_object_detections(np.load(clip_paths[n]))

        for n in range(N):
            vid = random.randint(1,100000000000) #generate a unique video id
            audio_path = clip_audio_paths[n]
            audio, audio_rate = librosa.load(audio_path, sr=self.opt.audio_sampling_rate)
            audio_segment= sample_audio(audio, self.audio_window)
            '''去掉音频部分的数据增强'''
            # if(self.opt.enable_data_augmentation and self.opt.mode == 'train'):
            #     audio_segment = augment_audio(audio_segment)
            audio_mag, audio_phase = generate_spectrogram_magphase(audio_segment, self.stft_frame, self.stft_hop)            
            detection_bbs = clip_det_bbs[n]
            audios[n] = audio_segment #make a copy of the audio to mix later

            if detection_bbs.shape[0] == 1:  # 如果检测到的乐器只有一个类别，则不处理，只用这一个
                frame_path = os.path.join(get_frame_root(clip_paths[n]), get_ins_name(clip_paths[n]),
                                          get_clip_name(clip_paths[n]),
                                          str(int(detection_bbs[0, 0])).zfill(6) + '.png')
                # todo: 这里修改了frame的获取路径，按照我自己处理数据的放置位置获取
                label = detection_bbs[0, 1] - 1  # make the label start from 0
                object_image = Image.open(frame_path).convert('RGB').crop(
                    (detection_bbs[0, -4], detection_bbs[0, -3], detection_bbs[0, -2], detection_bbs[0, -1]))

                if (self.opt.enable_data_augmentation and self.mode == 'train'):
                    object_image = augment_image(object_image)

                objects_visuals.append(self.vision_transform(object_image).unsqueeze(0))

                objects_labels.append(label)
                # make a copy of the audio spec for each object
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)

            elif detection_bbs.shape[0] >=2:  # 如果检测到的乐器类别数大于等于两个
                # 开始使用hand的信息对bboxes进行筛选
                hand_npy = get_hand_npy_path(clip_paths[n])  # 读取对应的hand检测结果
                if os.path.exists(hand_npy):        # 如果检测到hand，则读取检测结果
                    hand_bbs = np.load(hand_npy)
                else:
                    hand_bbs = np.array([])   # 如果对应的clip检测不到hand，则将hand的内容置为空

                if hand_bbs.shape[0] == 0:
                    hand_bb = np.array([])
                    sign = False
                    # print("this npy file {} donot have detected hands".format(os.path.basename(hand_npy)))
                elif hand_bbs.shape[0] == 1:  # 在检测到的乐器数不止一个的情况下，如果只检测到一只手，则就计算一只手的相关条件
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


                detection_bbs_filter = filter_det_bbs(hand_bb, sign, detection_bbs)

                frame_path = os.path.join(get_frame_root(clip_paths[n]), get_ins_name(clip_paths[n]),
                                              get_clip_name(clip_paths[n]),
                                              str(int(detection_bbs_filter[0])).zfill(6) + '.png')
                label = detection_bbs_filter[1] - 1  # make the label start from 0
                object_image = Image.open(frame_path).convert('RGB').crop(
                    (detection_bbs_filter[-4], detection_bbs_filter[-3], detection_bbs_filter[-2], detection_bbs_filter[-1]))

                if (self.opt.enable_data_augmentation and self.mode == 'train'):
                    object_image = augment_image(object_image)

                objects_visuals.append(self.vision_transform(object_image).unsqueeze(0))

                objects_labels.append(label)
                    # make a copy of the audio spec for each object
                objects_audio_mag.append(torch.FloatTensor(audio_mag).unsqueeze(0))
                objects_audio_phase.append(torch.FloatTensor(audio_phase).unsqueeze(0))
                objects_vids.append(vid)


            # audio_mix_mag += audio_mag
            # mix mag直接相加就可以，但是mix phase并不能通过相加得到。这里在训练时需要再思考mix的相位如何引入


        audio_mix = np.asarray(audios).sum(axis=0) # float(self.NUM_PER_MIX)
        audio_mix_mag, audio_mix_phase = generate_spectrogram_magphase(audio_mix, self.stft_frame, self.stft_hop)


        for n in range(self.NUM_PER_MIX):
            objects_audio_mix_mag.append(torch.FloatTensor(audio_mix_mag).unsqueeze(0))
            objects_audio_mix_phase.append(torch.FloatTensor(audio_mix_phase).unsqueeze(0))


        #stack all

        visuals = np.vstack(objects_visuals)  # detected objects
        audio_mags = np.vstack(objects_audio_mag)  # audio spectrogram magnitude
        audio_phases = np.vstack(objects_audio_phase)  # audio spectrogram phase
        labels = np.vstack(objects_labels)  # labels for each object, -1 denotes padded object
        vids = np.vstack(objects_vids)  # video indexes for each object, each video should have a unique id
        audio_mix_mags = np.vstack(objects_audio_mix_mag)
        audio_mix_phases = np.vstack(objects_audio_mix_phase)


        data = {'labels': labels, 'audio_mags': audio_mags, 'audio_mix_mags': audio_mix_mags, 'vids': vids}
        data['visuals'] = visuals

        if self.mode == 'val' or self.mode == 'test':
            data['audio_phases'] = audio_phases
            data['audio_mix_phases'] = audio_mix_phases
        return data

    def __len__(self):
        return len(self.list_sample)

    def name(self):
        return 'AudioVisualMUSICDataset'
