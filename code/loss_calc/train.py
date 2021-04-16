import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import math
import sys
import datetime
import time
from collections import namedtuple
from loss_calc.loss_main import get_coseparation_loss,get_coseparation_loss_multisource,get_refine_loss
from torch.autograd import Variable
import copy


def print_now(cmd, file=None):
    time_now = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    if file is None:
        print('%s %s' % (time_now, cmd))
    else:
        print_str = '%s %s' % (time_now, cmd)
        print(print_str, file=file)
    sys.stdout.flush()

def save_log_file(logger, loss, epoch, i, opt):
    # logger.info('-----iter:{}/{}-----'.format(i, opt.epoch_iters))
    t = 1
    keys = list(loss.keys())
    if len(keys)>0:
        logger.info('------Losses[{}]:[{}]/[{}]------'.format(epoch,i,opt.epoch_iters))
        for j in range(len(keys)):
            logger.info('{}_{}:{}'.format(t, keys[j], loss[keys[j]].data.item()))
            t = t+1
        logger.info('--------------------------------')

def to_tensorboard(writer, loss, i):
    keys = list(loss.keys())
    t = 1
    if len(keys) > 0:
        for j in range(len(keys)):
            writer.add_scalar('tb_results/{}_{}_G'.format(t,keys[j]), loss[keys[j]].data.item(), i)
            t = t + 1


def Train(opt, G, g_optimizer, data_loader, loss_set, epoch, logger, writer):

    loss_classification = loss_set['loss_classification']
    loss_coseparation = loss_set['loss_coseparation']
    loss_refine = loss_set['loss_refine']
    loss_audio_feature = loss_set['loss_audio']
    torch.set_grad_enabled(True)

    print('Training at {} epochs...'.format(epoch))

    G.train()
    torch.cuda.synchronize()

    for i, batch_data in enumerate(data_loader):
        G.zero_grad()
        g_optimizer.zero_grad()

        # gradients are accumulated through subiters
        g_output = G.forward(batch_data)

        '''1.classifier_loss '''
        classifier_loss = loss_classification(g_output['pred_label'], Variable(g_output['gt_label'],requires_grad=False)) * opt.classifier_loss_weight

        '''2.coseparation_loss and refine_loss'''
        # coseparation_loss, refine_loss = get_coseparation_loss(g_output, opt, loss_coseparation, loss_refine)
        coseparation_loss = get_coseparation_loss(g_output, opt, loss_coseparation)

        temp_output = {'vids':g_output['vids'], 'gt_mask': g_output['gt_mask'], 'weight': g_output['weight'], 'refine_mask': None}

        if opt.factor_type == 'down':
            factor = np.linspace(1,0,opt.refine_iteration+2)  # 等差数列降序
            factor = factor[1:-1]
            factor = factor/factor.sum()
            factor.tolist()
        elif opt.factor_type == 'up':
            factor = np.linspace(0, 1, opt.refine_iteration + 2)  # 等差数列升序
            factor = factor[1:-1]
            factor = factor / factor.sum()
            factor.tolist()
        else:
            factor = [1/opt.refine_iteration for n in range(opt.refine_iteration)]   # 系数均分

        if opt.refine_iteration == 1:
            temp_output['refine_mask'] = g_output['refine_masks'][0]
            refine_loss = get_refine_loss(temp_output,opt,loss_refine)
        else:
            temp_output['refine_mask'] = g_output['refine_masks'][0]
            refine_loss = factor[0] * get_refine_loss(temp_output, opt, loss_refine)
            for j in range(1,opt.refine_iteration):
                temp_output['refine_mask'] = g_output['refine_masks'][j]
                refine_loss += factor[j] *  get_refine_loss(temp_output, opt, loss_refine)

        '''3.audio feature loss'''
        loss_features = loss_audio_feature(g_output['real_audio_feat'], g_output['fake_audio_feat']) * 100

        loss_reconstruction = classifier_loss + coseparation_loss + refine_loss + loss_features


        loss_reconstruction.backward()
        g_optimizer.step()

        loss = {'loss_classifier':classifier_loss,'loss_coseparation': coseparation_loss}
        loss['loss_refine'] = refine_loss
        loss['audio_features'] = loss_features
        loss['loss_reconstruction'] = loss_reconstruction

        save_log_file(logger=logger, loss=loss, epoch=epoch, i=i, opt=opt)

        if opt.tensorboard:
            to_tensorboard(writer, loss, i)

