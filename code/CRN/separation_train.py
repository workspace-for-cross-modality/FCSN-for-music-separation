#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import time
from options.train_options import TrainOptions
from options.train_options_vggsound import vggTrainOptions
from data.data_loader import CreateDataLoader
from models.models import ModelBuilder
from models.audioVisual_model import AudioVisualModel
from scipy.misc import imsave
import scipy.io.wavfile as wavfile
import numpy as np
import torch
from torch.autograd import Variable
import librosa
from utils import utils,viz
from models import criterion
import torch.nn.functional as F
from torch import autograd
import random
from utils.utils import get_logger
from utils.functions_in_train import create_optimizer, decrease_learning_rate
from loss_calc.train import Train

from data.audioVisual_dataset import AudioVisualMUSICDataset
from data.audioVisual_vgg_dataset import AudioVisual_vggsound
from data.audioVisual_audioset import AudioVisual_audioset
import torch.utils.data
from utils.utils import object_collate


def load_checkpoint(nets, optimizer, filename):
    (net_visual, net_unet, net_classifier, net_refine, net_audio_extractor) = nets
    start_epoch = 1
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch'] + 1

        net_visual.load_state_dict(checkpoint['state_dict_visual'])
        net_unet.load_state_dict(checkpoint['state_dict_unet'])
        net_classifier.load_state_dict(checkpoint['state_dict_classifier'])
        net_refine.load_state_dict(checkpoint['state_dict_refine'])
        net_audio_extractor.load_state_dict(checkpoint['state_dict_down'])

        optimizer.load_state_dict(checkpoint['optimizer'])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.cuda()
        print("=> finish loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    nets = (net_visual, net_unet, net_classifier, net_refine, net_audio_extractor)
    return nets, optimizer, start_epoch

def save_checkpoint(nets, optimizer, epoch, opt):

    (net_visual, net_unet, net_classifier, net_refine, net_audio_extractor) = nets
    suffix_latest = 'latest.pth'
    suffix_best = 'best.pth'

    state = {'epoch': epoch,
             'state_dict_visual': net_visual.state_dict(),
             'state_dict_unet': net_unet.state_dict(),
             'state_dict_classifier': net_classifier.state_dict(),
             'state_dict_refine': net_refine.state_dict(),
             'state_dict_down': net_audio_extractor.state_dict(),
             'optimizer': optimizer.state_dict()}
    root = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(root):
        os.makedirs(root)
    torch.save(state, '{}/checkpoint_{}'.format(root, suffix_latest))


if __name__ == '__main__':

    os.environ["CUDA_VISIBLE_DEVICES"] = ""

    # parse arguments
    opt = TrainOptions().parse()
    opt.device = torch.device("cuda")
    opt.data_path = opt.data_path.replace('/home/', opt.hri_change)
    opt.data_path_duet = opt.data_path_duet.replace('/home/', opt.hri_change)
    opt.hdf5_path = opt.hdf5_path.replace('/home/', opt.hri_change)
    # 创建checkpoint和记录的文件夹
    path = os.path.join(opt.checkpoints_dir, opt.name)
    if not os.path.exists(path):
        os.makedirs(path)

    logger = get_logger(path)

    '''设置随机种子'''
    random.seed(opt.seed)
    np.random.seed(opt.seed)
    torch.manual_seed(opt.seed)
    torch.cuda.manual_seed(opt.seed)
    torch.cuda.manual_seed_all(opt.seed)

    ''' 建立数据读取迭代对象'''
    opt.batch_size = opt.batch_size_per_gpu * opt.num_gpus
    # dataset_train = AudioVisualMUSICDataset(mode='train', opt=opt)
    if opt.multisource:
        dataset_train = AudioVisual_soloduet(mode='train', opt=opt)
    elif opt.vggsound:
        dataset_train = AudioVisual_vggsound(mode='train', opt=opt)
    elif opt.audioset:
        dataset_train = AudioVisual_audioset(mode='train', opt=opt)
    elif opt.sameclass:
        dataset_train = AudioVisualMUSICDataset_same_class(mode='train', opt=opt)
    else:
        dataset_train = AudioVisualMUSICDataset(mode='train', opt=opt)
        # dataset_train = AudioVisualMUSICDataset_nohand(mode='train', opt=opt)


    loader_train = torch.utils.data.DataLoader(dataset_train,
                                               batch_size=opt.batch_size,
                                               shuffle=True,
                                               num_workers=opt.nThreads,
                                               drop_last=True,
                                               collate_fn=object_collate)

    opt.epoch_iters = len(dataset_train) // opt.batch_size
    print('1 Epoch = {} iters'.format(opt.epoch_iters))

    if opt.tensorboard:
        from tensorboardX import SummaryWriter
        writer = SummaryWriter(comment=opt.name)
    else:
        writer = None

    ''' 模型建立'''
    builder = ModelBuilder()
    net_visual = builder.build_visual(
        pool_type=opt.visual_pool,
        fc_out=512,
        weights=opt.weights_visual)
    net_unet = builder.build_unet(
        unet_num_layers=opt.unet_num_layers,
        ngf=opt.unet_ngf,
        input_nc=opt.unet_input_nc,
        output_nc=opt.unet_output_nc,
        weights=opt.weights_unet)
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

    g_optimizer = create_optimizer(nets, opt)

    start_epoch = 1

    ''' 训练中断时继续训练'''
    if opt.continue_train:
        model_name = os.path.join(opt.checkpoints_dir, opt.name, 'checkpoint_latest.pth')
        nets, g_optimizer, start_epoch = load_checkpoint(nets,g_optimizer,filename=model_name)
        print('continue train at epoch {}'.format(start_epoch))

    G = AudioVisualModel(nets, opt)

    ''' 设置网络的loss function'''
    loss_classification = criterion.CELoss()
    if opt.mask_loss_type == 'L1':
        loss_coseparation = criterion.L1Loss()
        loss_refine = criterion.L1Loss()
    elif opt.mask_loss_type == 'L2':
        loss_coseparation = criterion.L2Loss()
        loss_refine = criterion.L2Loss()
    elif opt.mask_loss_type == 'BCE':
        loss_coseparation = criterion.BCELoss()
        loss_refine = criterion.BCELoss()
    loss_audio_feature = torch.nn.MSELoss(reduce=True, size_average=True)

    ''' 设置网络的并行计算'''
    if (opt.num_gpus > 0):
        loss_classification.cuda()
        loss_coseparation.cuda()
        loss_refine.cuda()
        loss_audio_feature.cuda()

        G = torch.nn.DataParallel(G, device_ids=range(opt.num_gpus)).cuda()
        G.to(opt.device)

    loss_set = {'loss_classification': loss_classification, 'loss_coseparation': loss_coseparation,
                'loss_refine': loss_refine, 'loss_audio': loss_audio_feature}

    # initialization
    batch_classifier_loss = []
    batch_coseparation_loss = []
    batch_refine_loss = []
    best_err = float("inf")


    ''' 训练迭代开始'''
    for epoch in range(start_epoch, opt.num_epoch+1):
        Train(opt, G, g_optimizer, loader_train, loss_set, epoch, logger, writer)
        # 保存模型结果
        if (epoch % opt.save_latest_freq == 0):
            logger.info('saving the latest model [{}/{}]'.format(epoch, opt.num_epoch))
            save_checkpoint(nets, g_optimizer, epoch, opt)
            logger.info('models sucessfully saved (iter:{})'.format(epoch))

        if (epoch in opt.lr_steps):
            decrease_learning_rate(g_optimizer, opt.decay_factor)
            print('decreased learning rate by ', opt.decay_factor)

    if writer:
        writer.close()

