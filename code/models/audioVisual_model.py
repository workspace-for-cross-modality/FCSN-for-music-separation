import numpy as np
import torch
import os
from torch import optim
import torch.nn.functional as F
from . import networks,criterion
from utils.utils import warpgrid
from torch.autograd import Variable
import copy
from loss_calc.loss_main import get_coseparation_loss,get_coseparation_loss_multisource,get_refine_loss

class AudioVisualModel(torch.nn.Module):
    def name(self):
        return 'AudioVisualModel'

    def __init__(self, nets, opt):
        super(AudioVisualModel, self).__init__()
        self.opt = opt

        #initialize model and criterions
        self.net_visual, self.net_unet, self.net_classifier, self.net_refine, self.audio_extractor = nets

    def forward(self, input):
        labels = input['labels']
        labels = labels.squeeze(1).long() #covert back to longtensor
        vids = input['vids']
        audio_mags = input['audio_mags']
        audio_mix_mags = input['audio_mix_mags']
        visuals = input['visuals']
        # visuals_256 = input['visuals_256']

        audio_mix_mags = audio_mix_mags + 1e-10

        '''1. warp the spectrogram'''
        B = audio_mix_mags.size(0)
        T = audio_mix_mags.size(3)
        if self.opt.log_freq:
            grid_warp = torch.from_numpy(warpgrid(B, 256, T, warp=True)).to(self.opt.device)
            audio_mix_mags = F.grid_sample(audio_mix_mags, grid_warp)
            audio_mags = F.grid_sample(audio_mags, grid_warp)

        '''2. calculate ground-truth masks'''
        gt_masks = audio_mags / audio_mix_mags
        # clamp to avoid large numbers in ratio masks
        gt_masks.clamp_(0., 5.)

        '''3. pass through visual stream and extract visual features'''
        visual_feature, _ = self.net_visual(Variable(visuals, requires_grad=False))

        '''4. audio-visual feature fusion through UNet and predict mask'''
        audio_log_mags = torch.log(audio_mix_mags).detach()

        # audio_norm_mags = torch.sigmoid(torch.log(audio_mags + 1e-10))


        mask_prediction = self.net_unet(audio_log_mags, visual_feature)

        '''5. masking the spectrogram of mixed audio to perform separation and predict classification label'''
        separated_spectrogram = audio_mix_mags * mask_prediction

        # generate spectrogram for the classifier
        spectrogram2classify = torch.log(separated_spectrogram + 1e-10)  # get log spectrogram

        # calculate loss weighting coefficient
        if self.opt.weighted_loss:
            weight = torch.log1p(audio_mix_mags)
            weight = torch.clamp(weight, 1e-3, 10)
        else:
            weight = None

        ''' 6.classify the predicted spectrogram'''
        ''' add audio feature after resnet18 layer4, 512*8*8'''
        ''' add output for classifier, output:label,feature(after layer4)'''
        label_prediction, _ = self.net_classifier(spectrogram2classify)


        # if self.opt.visual_unet_encoder:
        #     refine_mask, left_mask = self.refine_iteration(mask_prediction, audio_mix_mags, None) #visuals_256)
        # elif self.opt.visual_cat:
        #     refine_mask, left_mask = self.refine_iteration(mask_prediction, audio_mix_mags, visual_feature)
        # else:
        #     refine_mask, left_mask = self.refine_iteration(mask_prediction, audio_mix_mags, None)
        refine_masks = [None for i in range(self.opt.refine_iteration)]
        temp_mask = mask_prediction
        left_energy = [None for i in range(self.opt.refine_iteration)]
        for i in range(self.opt.refine_iteration):
            refine_mask, left_mask , left_mags = self.refine_iteration(temp_mask, audio_mix_mags, visual_feature)
            refine_masks[i] = refine_mask
            temp_mask = refine_mask
            left_energy[i] = torch.mean(left_mags)




        # refine后的频谱
        refine_spec = audio_mix_mags * refine_mask
        # refine_norm_mags = torch.sigmoid(torch.log(refine_spec + 1e-10))

        refine2classify = torch.log(refine_spec + 1e-10)
        _, fake_audio_feature = self.net_classifier(refine2classify)

        ''' 7. down channels for audio feature, for cal loss'''
        if self.opt.audio_extractor:
            real_audio_mags = torch.log(audio_mags + 1e-10)
            _ ,real_audio_feature = self.net_classifier(real_audio_mags)
            real_audio_feature = self.audio_extractor(real_audio_feature)
            fake_audio_feature = self.audio_extractor(fake_audio_feature)


        output = {'gt_label': labels, 'pred_label': label_prediction, 'pred_mask': mask_prediction, 'gt_mask': gt_masks,
                'pred_spectrogram': separated_spectrogram, 'visual_object': visuals, 'audio_mags': audio_mags,
                  'audio_mix_mags': audio_mix_mags, 'weight': weight, 'vids': vids,
                  'refine_mask': refine_mask, 'refine_spec': refine_spec, 'left_mask':left_mask, 'refine_masks':refine_masks,
                  'left_mags': left_mags, 'left_energy':left_energy}
        if self.opt.audio_extractor:
            output['real_audio_feat'] = real_audio_feature
            output['fake_audio_feat'] = fake_audio_feature
        return output



    def refine_iteration(self, mask_in, audio_mix_mags, visual_feature):
        separated_spec = mask_in * audio_mix_mags
        mask = mask_in
        # for i in range(self.opt.refine_iteration):
        start = 0
        end = self.opt.num_per_mix
        audio_average_mags = torch.zeros(audio_mix_mags.shape).cuda()
        for j in range(self.opt.batch_size):
            '''mix mags already * 2, only sum is ok, no average'''
            # solo = separated_spec[start]
            # duet = torch.sum(separated_spec[start+1:end],dim=0)/2
            audio_average_mags[start:end] = torch.sum(separated_spec[start:end], dim=0)  #/ float(self.opt.num_per_mix)
            start = copy.deepcopy(end)
            end = end + self.opt.num_per_mix
        left_mags = audio_mix_mags - audio_average_mags
        left_mask = left_mags / audio_mix_mags

        refine_in = torch.cat((mask, left_mask), dim=1)
        mask = self.net_refine(refine_in, visual_feature)

        # separated_spec = mask * audio_mix_mags

        return mask, left_mask, left_mags




        # start = 0
        # end = 2
        # audio_average_mags = torch.zeros(audio_mix_mags.shape).cuda()
        # for i in range(self.opt.batchSize):
        #     '''mix mags already * 2, only sum is ok, no average'''
        #     audio_average_mags[start:end] = torch.sum(separated_spectrogram[start:end], dim=0)
        #     start = copy.deepcopy(end)
        #     end = end + 2
        # left_mags = audio_mix_mags - audio_average_mags
        # left_mask = left_mags / audio_mix_mags
        #
        # # refine unet7 input spectrogram
        # refine_in = torch.cat((mask_prediction, left_mask), dim=1)
        #
        # if self.opt.visual_unet_encoder:
        #     refine_mask = self.net_refine(refine_in, visuals_256)
        # elif self.opt.visual_cat:
        #     refine_mask = self.net_refine(refine_in, visual_feature)
        # else:
        #     refine_mask = self.net_refine(refine_in)

