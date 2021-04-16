import torch
import copy


def get_coseparation_loss(output, opt, loss_coseparation):
    # initialize a dic to store the index of the list
    vid_index_dic = {}
    vids = output['vids'].squeeze(1).cpu().numpy()
    O = vids.shape[0]
    count = 0
    for i in range(O):
        if not vid_index_dic.__contains__(vids[i]):
            vid_index_dic[vids[i]] = count
            count = count + 1

    # initialize three lists of length = number of video clips to reconstruct
    predicted_mask_list = [None for i in range(len(vid_index_dic.keys()))]
    gt_mask_list = [None for i in range(len(vid_index_dic.keys()))]
    weight_list = [None for i in range(len(vid_index_dic.keys()))]
    # refine_mask_list = [None for i in range(len(vid_index_dic.keys()))]
    # audio_mix_mags_list = [None for i in range(len(vid_index_dic.keys()))]

    # iterate through all objects
    gt_masks = output['gt_mask']
    mask_prediction = output['pred_mask']
    weight = output['weight']
    # refine_mask = output['refine_mask']
    # audio_mags = output['audio_mags']

    for i in range(O):
        if predicted_mask_list[vid_index_dic[vids[i]]] is None:
            gt_mask_list[vid_index_dic[vids[i]]] = gt_masks[i, :, :, :]
            weight_list[vid_index_dic[vids[i]]] = weight[i, :, :, :]
            predicted_mask_list[vid_index_dic[vids[i]]] = mask_prediction[i, :, :, :]
            # refine_mask_list[vid_index_dic[vids[i]]] = refine_mask[i, :, :, :]
            # audio_mix_mags_list[vid_index_dic[vids[i]]] = audio_mix_mags[i,:,:,:]
        else:
            predicted_mask_list[vid_index_dic[vids[i]]] = predicted_mask_list[vid_index_dic[vids[i]]] + mask_prediction[i, :, :, :]
            # refine_mask_list[vid_index_dic[vids[i]]] = refine_mask_list[vid_index_dic[vids[i]]] + refine_mask[i, :, :, :]

    if opt.mask_loss_type == 'BCE':
        for i in range(O):
            # clip the prediction results to make it in the range of [0,1] for BCE loss
            predicted_mask_list[vid_index_dic[vids[i]]] = torch.clamp(predicted_mask_list[vid_index_dic[vids[i]]], 0, 1)
    coseparation_loss = loss_coseparation(predicted_mask_list, gt_mask_list, weight_list)
    # refine_loss = loss_refine(refine_mask_list, gt_mask_list, weight_list)
    return coseparation_loss

def get_refine_loss(output, opt, loss_refine):
    # initialize a dic to store the index of the list
    vid_index_dic = {}
    vids = output['vids'].squeeze(1).cpu().numpy()
    O = vids.shape[0]
    count = 0
    for i in range(O):
        if not vid_index_dic.__contains__(vids[i]):
            vid_index_dic[vids[i]] = count
            count = count + 1

    # initialize three lists of length = number of video clips to reconstruct

    gt_mask_list = [None for i in range(len(vid_index_dic.keys()))]
    weight_list = [None for i in range(len(vid_index_dic.keys()))]
    refine_mask_list = [None for i in range(len(vid_index_dic.keys()))]
    # audio_mix_mags_list = [None for i in range(len(vid_index_dic.keys()))]

    # iterate through all objects
    gt_masks = output['gt_mask']
    weight = output['weight']
    refine_mask = output['refine_mask']
    # audio_mags = output['audio_mags']

    for i in range(O):
        if refine_mask_list[vid_index_dic[vids[i]]] is None:
            gt_mask_list[vid_index_dic[vids[i]]] = gt_masks[i, :, :, :]
            weight_list[vid_index_dic[vids[i]]] = weight[i, :, :, :]
            refine_mask_list[vid_index_dic[vids[i]]] = refine_mask[i, :, :, :]
            # audio_mix_mags_list[vid_index_dic[vids[i]]] = audio_mix_mags[i,:,:,:]
        else:
            refine_mask_list[vid_index_dic[vids[i]]] = refine_mask_list[vid_index_dic[vids[i]]] + refine_mask[i, :, :, :]

    if opt.mask_loss_type == 'BCE':
        for i in range(O):
            # clip the prediction results to make it in the range of [0,1] for BCE loss
            refine_mask_list[vid_index_dic[vids[i]]] = torch.clamp(refine_mask_list[vid_index_dic[vids[i]]], 0, 1)
    refine_loss = loss_refine(refine_mask_list, gt_mask_list, weight_list)
    return refine_loss






