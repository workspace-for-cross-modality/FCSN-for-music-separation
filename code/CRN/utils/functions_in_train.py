
import os

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
from loss_calc.loss_main import get_coseparation_loss


def create_optimizer(nets, opt):
    (net_visual, net_unet, net_classifier, net_refine, net_audio_ext) = nets
    param_groups = [{'params': net_visual.parameters(), 'lr': opt.lr_visual},
                    {'params': net_unet.parameters(), 'lr': opt.lr_unet},
                    {'params': net_classifier.parameters(), 'lr': opt.lr_classifier},
                    {'params': net_refine.parameters(), 'lr': opt.lr_refine},
                    {'params': net_audio_ext.parameters(), 'lr': opt.lr_refine}]
    if opt.optimizer == 'sgd':
        return torch.optim.SGD(param_groups, momentum=opt.beta1, weight_decay=opt.weight_decay)
    elif opt.optimizer == 'adam':
        return torch.optim.Adam(param_groups, betas=(opt.beta1, 0.999), weight_decay=opt.weight_decay)


def decrease_learning_rate(optimizer, decay_factor=0.1):
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay_factor


def save_visualization(vis_rows, outputs, batch_data, save_dir, opt):
    # fetch data and predictions
    mag_mix = batch_data['audio_mix_mags']
    phase_mix = batch_data['audio_mix_phases']
    visuals = batch_data['visuals']

    pred_masks_ = outputs['pred_mask']
    gt_masks_ = outputs['gt_mask']
    mag_mix_ = outputs['audio_mix_mags']
    weight_ = outputs['weight']
    visual_object = outputs['visual_object']
    gt_label = outputs['gt_label']
    _, pred_label = torch.max(outputs['pred_label'], 1)
    label_list = ['Banjo', 'Cello', 'Drum', 'Guitar', 'Harp', 'Harmonica', 'Oboe', 'Piano', 'Saxophone', \
                  'Trombone', 'Trumpet', 'Violin', 'Flute', 'Accordion', 'Horn']

    # unwarp log scale
    B = mag_mix.size(0)
    if opt.log_freq:
        grid_unwarp = torch.from_numpy(utils.warpgrid(B, opt.stft_frame // 2 + 1, gt_masks_.size(3), warp=False)).to(
            opt.device)
        pred_masks_linear = F.grid_sample(pred_masks_, grid_unwarp)
        gt_masks_linear = F.grid_sample(gt_masks_, grid_unwarp)
    else:
        pred_masks_linear = pred_masks_
        gt_masks_linear = gt_masks_

    # convert into numpy
    mag_mix = mag_mix.numpy()
    mag_mix_ = mag_mix_.detach().cpu().numpy()
    phase_mix = phase_mix.numpy()
    weight_ = weight_.detach().cpu().numpy()
    pred_masks_ = pred_masks_.detach().cpu().numpy()
    pred_masks_linear = pred_masks_linear.detach().cpu().numpy()
    gt_masks_ = gt_masks_.detach().cpu().numpy()
    gt_masks_linear = gt_masks_linear.detach().cpu().numpy()
    visual_object = visual_object.detach().cpu().numpy()
    gt_label = gt_label.detach().cpu().numpy()
    pred_label = pred_label.detach().cpu().numpy()

    # loop over each example
    for j in range(min(B, opt.num_visualization_examples)):
        row_elements = []

        # video names
        prefix = str(j) + '-' + label_list[int(gt_label[j])] + '-' + label_list[int(pred_label[j])]
        utils.mkdirs(os.path.join(save_dir, prefix))

        # save mixture
        mix_wav = utils.istft_coseparation(mag_mix[j, 0], phase_mix[j, 0], hop_length=opt.stft_hop)
        mix_amp = utils.magnitude2heatmap(mag_mix_[j, 0])
        weight = utils.magnitude2heatmap(weight_[j, 0], log=False, scale=100.)
        filename_mixwav = os.path.join(prefix, 'mix.wav')
        filename_mixmag = os.path.join(prefix, 'mix.jpg')
        filename_weight = os.path.join(prefix, 'weight.jpg')
        imsave(os.path.join(save_dir, filename_mixmag), mix_amp[::-1, :, :])
        imsave(os.path.join(save_dir, filename_weight), weight[::-1, :])
        wavfile.write(os.path.join(save_dir, filename_mixwav), opt.audio_sampling_rate, mix_wav)
        row_elements += [{'text': prefix}, {'image': filename_mixmag, 'audio': filename_mixwav}]

        # GT and predicted audio reconstruction
        gt_mag = mag_mix[j, 0] * gt_masks_linear[j, 0]
        gt_wav = utils.istft_coseparation(gt_mag, phase_mix[j, 0], hop_length=opt.stft_hop)
        pred_mag = mag_mix[j, 0] * pred_masks_linear[j, 0]
        preds_wav = utils.istft_coseparation(pred_mag, phase_mix[j, 0], hop_length=opt.stft_hop)

        # output masks
        filename_gtmask = os.path.join(prefix, 'gtmask.jpg')
        filename_predmask = os.path.join(prefix, 'predmask.jpg')
        gt_mask = (np.clip(gt_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        pred_mask = (np.clip(pred_masks_[j, 0], 0, 1) * 255).astype(np.uint8)
        imsave(os.path.join(save_dir, filename_gtmask), gt_mask[::-1, :])
        imsave(os.path.join(save_dir, filename_predmask), pred_mask[::-1, :])

        # ouput spectrogram (log of magnitude, show colormap)
        filename_gtmag = os.path.join(prefix, 'gtamp.jpg')
        filename_predmag = os.path.join(prefix, 'predamp.jpg')
        gt_mag = utils.magnitude2heatmap(gt_mag)
        pred_mag = utils.magnitude2heatmap(pred_mag)
        imsave(os.path.join(save_dir, filename_gtmag), gt_mag[::-1, :, :])
        imsave(os.path.join(save_dir, filename_predmag), pred_mag[::-1, :, :])

        # output audio
        filename_gtwav = os.path.join(prefix, 'gt.wav')
        filename_predwav = os.path.join(prefix, 'pred.wav')
        wavfile.write(os.path.join(save_dir, filename_gtwav), opt.audio_sampling_rate, gt_wav)
        wavfile.write(os.path.join(save_dir, filename_predwav), opt.audio_sampling_rate, preds_wav)

        row_elements += [
            {'image': filename_predmag, 'audio': filename_predwav},
            {'image': filename_gtmag, 'audio': filename_gtwav},
            {'image': filename_predmask},
            {'image': filename_gtmask}]

        row_elements += [{'image': filename_weight}]
        vis_rows.append(row_elements)


# used to display validation loss
def display_val(model, crit, writer, index, dataset_val, opt):
    # remove previous viz results
    save_dir = os.path.join('.', opt.checkpoints_dir, opt.name, 'visualization')
    utils.mkdirs(save_dir)

    # initial results lists
    accuracies = []
    classifier_losses = []
    coseparation_losses = []

    # initialize HTML header
    visualizer = viz.HTMLVisualizer(os.path.join(save_dir, 'index.html'))
    header = ['Filename', 'Input Mixed Audio']
    header += ['Predicted Audio' 'GroundTruth Audio', 'Predicted Mask', 'GroundTruth Mask', 'Loss weighting']
    visualizer.add_header(header)
    vis_rows = []

    with torch.no_grad():
        for i, val_data in enumerate(dataset_val):
            if i < opt.validation_batches:
                output = model.forward(val_data)
                loss_classification = crit['loss_classification']
                classifier_loss = loss_classification(output['pred_label'], Variable(output['gt_label'],
                                                                                     requires_grad=False)) * opt.classifier_loss_weight
                coseparation_loss = get_coseparation_loss(output, opt, crit['loss_coseparation'])
                classifier_losses.append(classifier_loss.item())
                coseparation_losses.append(coseparation_loss.item())
                gt_label = output['gt_label']
                _, pred_label = torch.max(output['pred_label'], 1)
                accuracy = torch.sum(gt_label == pred_label).item() * 1.0 / pred_label.shape[0]
                accuracies.append(accuracy)
            else:
                if opt.validation_visualization:
                    output = model.forward(val_data)
                    save_visualization(vis_rows, output, val_data, save_dir, opt)  # visualize one batch
                break

    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_classifier_loss = sum(classifier_losses) / len(classifier_losses)
    avg_coseparation_loss = sum(coseparation_losses) / len(coseparation_losses)
    if opt.tensorboard:
        writer.add_scalar('data/val_classifier_loss', avg_classifier_loss, index)
        writer.add_scalar('data/val_accuracy', avg_accuracy, index)
        writer.add_scalar('data/val_coseparation_loss', avg_coseparation_loss, index)
    print('val accuracy: %.3f' % avg_accuracy)
    print('val classifier loss: %.3f' % avg_classifier_loss)
    print('val coseparation loss: %.3f' % avg_coseparation_loss)
    return avg_coseparation_loss + avg_classifier_loss