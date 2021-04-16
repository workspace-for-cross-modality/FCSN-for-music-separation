import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import functools
from torch.nn.utils import spectral_norm as spectral_norm_fn
from torch.nn.utils import weight_norm as weight_norm_fn
import torch.nn.utils.spectral_norm as spectral_norm

def unet_conv(input_nc, output_nc, norm_layer=nn.BatchNorm2d):
    downconv = nn.Conv2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    downrelu = nn.LeakyReLU(0.2, True)
    downnorm = norm_layer(output_nc)
    return nn.Sequential(*[downconv, downnorm, downrelu])

def unet_upconv(input_nc, output_nc, outermost=False, norm_layer=nn.BatchNorm2d):
    upconv = nn.ConvTranspose2d(input_nc, output_nc, kernel_size=4, stride=2, padding=1)
    uprelu = nn.ReLU(True)
    upnorm = norm_layer(output_nc)
    if not outermost:
        return nn.Sequential(*[upconv, upnorm, uprelu])
    else:
        return nn.Sequential(*[upconv, nn.Sigmoid()])
        
def create_conv(input_channels, output_channels, kernel, paddings, batch_norm=True, Relu=True, stride=1):
    model = [nn.Conv2d(input_channels, output_channels, kernel, stride = stride, padding = paddings)]
    if(batch_norm):
        model.append(nn.BatchNorm2d(output_channels))

    if(Relu):
        model.append(nn.ReLU())

    return nn.Sequential(*model)

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)
    elif classname.find('Linear') != -1:
        m.weight.data.normal_(0.0, 0.02)

'''
audio extrator, after resnet18 layer4, and down channels
'''
class Down_channels(nn.Module):
    def __init__(self):
        super(Down_channels, self).__init__()
        self.conv1x1 = create_conv(512, 128, 1, 0)
        self.conv1x1.apply(weights_init)
        self.fc = nn.Linear(8192, 512)
        self.fc.apply(weights_init)

    def forward(self, x):
        x = self.conv1x1(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = x.view(x.size(0), -1, 1, 1)
        return x

# class Down_channels(nn.Module):
#     def __init__(self):
#         super(Down_channels, self).__init__()
#         self.conv1_1 = nn.Conv2d(512,256, 1,padding=0)
#         self.conv1_2 = nn.Conv2d(256, 128, 1, padding=0)
#         self.conv1_3 = nn.Conv2d(128, 64, 1, padding=0)
#         self.conv1_4 = nn.Conv2d(64, 32, 1, padding=0)
#         # self.conv1_5 = nn.Conv2d(32, 16, 1, padding=0)
#         # self.conv1_6 = nn.Conv2d(16, 2, 1, padding=0)
#
#     def forward(self, x):
#         x = self.conv1_1(x)
#         x = self.conv1_2(x)
#         x = self.conv1_3(x)
#         x = self.conv1_4(x)
#         # x = self.conv1_5(x)
#         # x = self.conv1_6(x)
#         return x



class Resnet18(nn.Module):
    def __init__(self, original_resnet, pool_type='maxpool', input_channel=3, with_fc=False, fc_in=512, fc_out=512):
        super(Resnet18, self).__init__()
        self.pool_type = pool_type
        self.input_channel = input_channel
        self.with_fc = with_fc

        #customize first convolution layer to handle different number of channels for images and spectrograms
        self.conv1 = nn.Conv2d(self.input_channel, 64, kernel_size=7, stride=2, padding=3, bias=False)
        layers = [self.conv1]
        layers.extend(list(original_resnet.children())[1:-2])
        self.feature_extraction = nn.Sequential(*layers) #features before pooling
        #print self.feature_extraction

        if pool_type == 'conv1x1':
            self.conv1x1 = create_conv(512, 128, 1, 0)
            self.conv1x1.apply(weights_init)

        if with_fc:
            self.fc = nn.Linear(fc_in, fc_out)
            self.fc.apply(weights_init)

    def forward(self, x):
        feat = self.feature_extraction(x)

        if self.pool_type == 'avgpool':
            x = F.adaptive_avg_pool2d(feat, 1)
        elif self.pool_type == 'maxpool':
            x = F.adaptive_max_pool2d(feat, 1)
        elif self.pool_type == 'conv1x1':
            x = self.conv1x1(feat)
            # print('shape after 1*1:{}'.format(x.size()))
            # import numpy as np
            # tem_visual = x[0,0,:,:]
            # tem_visual = torch.nn.functional.sigmoid(tem_visual)
            # tem_visual = tem_visual.detach().cpu().numpy()
            # tem_visual = np.round(tem_visual * 255)
            # import cv2
            # cv2.imwrite('/home/mashuo/work/study/co-separation/test_for_visualfea.png', tem_visual)
            # print('success save visual feature')

        else:
            return feat  # no pooling and conv1x1, directly return the feature map

        if self.with_fc:
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            if self.pool_type == 'conv1x1':
                x = x.view(x.size(0), -1, 1, 1) #expand dimension if using conv1x1 + fc to reduce dimension
            return x, feat
        else:
            return x, feat

class AudioVisual7layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual7layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)    # _×1×256×256 -> _×64×128×128
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)  # _×128×64×64
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)  # _×256×32×32
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)  # _×512×16×16
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)  # _×512×8×8
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)  # _×512×4×4
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)  # _×512×2×2

        visual_feat = visual_feat.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
        audioVisual_feature = torch.cat((visual_feat, audio_conv7feature), dim=1)   #_ 1024.2.2

        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)  # _.512.4.4
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))  # _.512.8.8
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))  # _.512.16.16
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))  # _.256.32.32
        audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))  # _.128.64.64
        audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))  # _.64.128.128
        mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))       # _.1.256.256
        return mask_prediction

class AudioVisual5layerUNet(nn.Module):
    def __init__(self, ngf=64, input_nc=2, output_nc=2):
        super(AudioVisual5layerUNet, self).__init__()

        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

    def forward(self, x, visual_feat):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)

        visual_feat = visual_feat.repeat(1, 1, audio_conv5feature.shape[2], audio_conv5feature.shape[3])
        audioVisual_feature = torch.cat((visual_feat, audio_conv5feature), dim=1)
        audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
        audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv4feature), dim=1))
        audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv3feature), dim=1))
        audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv2feature), dim=1))
        mask_prediction = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv1feature), dim=1))
        return mask_prediction


class Normal_UNet7(nn.Module): # normal U-net 7 , refine net
    def __init__(self, opt, ngf=64, input_nc=2, output_nc=1, vis_input_nc=3):
        super(Normal_UNet7, self).__init__()
        self.opt = opt
        #initialize layers
        self.audionet_convlayer1 = unet_conv(input_nc, ngf)
        self.audionet_convlayer2 = unet_conv(ngf, ngf * 2)
        self.audionet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
        self.audionet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
        self.audionet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
        self.audionet_convlayer7 = unet_conv(ngf * 8, ngf * 8)

        if not self.opt.visual_unet_encoder and not self.opt.visual_cat:
            self.audionet_upconvlayer1 = unet_upconv(ngf * 8, ngf * 8)
        else:
            self.audionet_upconvlayer1 = unet_upconv(ngf * 16, ngf * 8)    # without visual:8,8, with visual:16,8
        self.audionet_upconvlayer2 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer3 = unet_upconv(ngf * 16, ngf * 8)
        self.audionet_upconvlayer4 = unet_upconv(ngf * 16, ngf *4)
        self.audionet_upconvlayer5 = unet_upconv(ngf * 8, ngf * 2)
        self.audionet_upconvlayer6 = unet_upconv(ngf * 4, ngf)
        self.audionet_upconvlayer7 = unet_upconv(ngf * 2, output_nc, True) #outermost layer use a sigmoid to bound the mask

        if self.opt.visual_unet_encoder:
            # visual encoder used images
            self.visualnet_convlayer1 = unet_conv(vis_input_nc, ngf)
            self.visualnet_convlayer2 = unet_conv(ngf, ngf * 2)
            self.visualnet_convlayer3 = unet_conv(ngf * 2, ngf * 4)
            self.visualnet_convlayer4 = unet_conv(ngf * 4, ngf * 8)
            self.visualnet_convlayer5 = unet_conv(ngf * 8, ngf * 8)
            self.visualnet_convlayer6 = unet_conv(ngf * 8, ngf * 8)
            self.visualnet_convlayer7 = unet_conv(ngf * 8, ngf * 8)


    def forward(self, x, visual=None):
        audio_conv1feature = self.audionet_convlayer1(x)
        audio_conv2feature = self.audionet_convlayer2(audio_conv1feature)
        audio_conv3feature = self.audionet_convlayer3(audio_conv2feature)
        audio_conv4feature = self.audionet_convlayer4(audio_conv3feature)
        audio_conv5feature = self.audionet_convlayer5(audio_conv4feature)
        audio_conv6feature = self.audionet_convlayer6(audio_conv5feature)
        audio_conv7feature = self.audionet_convlayer7(audio_conv6feature)

        if self.opt.visual_unet_encoder:
            vis_conv1feature = self.visualnet_convlayer1(visual)
            vis_conv2feature = self.visualnet_convlayer2(vis_conv1feature)
            vis_conv3feature = self.visualnet_convlayer3(vis_conv2feature)
            vis_conv4feature = self.visualnet_convlayer4(vis_conv3feature)
            vis_conv5feature = self.visualnet_convlayer5(vis_conv4feature)
            vis_conv6feature = self.visualnet_convlayer6(vis_conv5feature)
            vis_conv7feature = self.visualnet_convlayer7(vis_conv6feature)

            audioVisual_feature = torch.cat((vis_conv7feature, audio_conv7feature), dim=1)  # _ 1024.2.2

            audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)
            audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, vis_conv6feature), dim=1))
            audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, vis_conv5feature), dim=1))
            audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, vis_conv4feature), dim=1))
            audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, vis_conv3feature), dim=1))
            audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, vis_conv2feature), dim=1))
            mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, vis_conv1feature), dim=1))

        elif self.opt.visual_cat:
            visual_feat = visual.repeat(1, 1, audio_conv7feature.shape[2], audio_conv7feature.shape[3])
            audioVisual_feature = torch.cat((visual_feat, audio_conv7feature), dim=1)  # _ 1024.2.2

            audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)  # _.512.4.4
            audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))  # _.512.8.8
            audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))  # _.512.16.16
            audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))  # _.256.32.32
            audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))  # _.128.64.64
            audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))  # _.64.128.128
            mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))  # _.1.256.256
        else:
            audioVisual_feature = audio_conv7feature  # 暂时按此统一命名，实际只有audio feature引入

            audio_upconv1feature = self.audionet_upconvlayer1(audioVisual_feature)  # _.512.4.4
            audio_upconv2feature = self.audionet_upconvlayer2(torch.cat((audio_upconv1feature, audio_conv6feature), dim=1))  # _.512.8.8
            audio_upconv3feature = self.audionet_upconvlayer3(torch.cat((audio_upconv2feature, audio_conv5feature), dim=1))  # _.512.16.16
            audio_upconv4feature = self.audionet_upconvlayer4(torch.cat((audio_upconv3feature, audio_conv4feature), dim=1))  # _.256.32.32
            audio_upconv5feature = self.audionet_upconvlayer5(torch.cat((audio_upconv4feature, audio_conv3feature), dim=1))  # _.128.64.64
            audio_upconv6feature = self.audionet_upconvlayer6(torch.cat((audio_upconv5feature, audio_conv2feature), dim=1))  # _.64.128.128
            mask_prediction = self.audionet_upconvlayer7(torch.cat((audio_upconv6feature, audio_conv1feature), dim=1))  # _.1.256.256

        return mask_prediction

class D(nn.Module):
    def __init__(self, input_dim=3, ndf=64, use_cuda=True, device_ids=None):
        super(D, self).__init__()
        self.input_dim = input_dim
        self.cnum = ndf
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.dis_conv_module = DisConvModule(self.input_dim, self.cnum)
        self.linear = nn.Linear(self.cnum*4*16*16, 1)

    def forward(self, x):   # x:256*256
        x = self.dis_conv_module(x)
        x = x.view(x.size()[0], -1)
        x = self.linear(x)

        return x


class DisConvModule(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(DisConvModule, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids

        self.conv1 = dis_conv(input_dim, cnum, 5, 2, 2)
        self.conv2 = dis_conv(cnum, cnum*2, 5, 2, 2)
        self.conv3 = dis_conv(cnum*2, cnum*4, 5, 2, 2)
        self.conv4 = dis_conv(cnum*4, cnum*4, 5, 2, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        return x


def dis_conv(input_dim, output_dim, kernel_size=5, stride=2, padding=0, rate=1,
             activation='lrelu'):
    return Conv2dBlock(input_dim, output_dim, kernel_size, stride,
                       conv_padding=padding, dilation=rate,
                       activation=activation)


class Conv2dBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0,
                 conv_padding=0, dilation=1, weight_norm='none', norm='none',
                 activation='relu', pad_type='zero', transpose=False):
        super(Conv2dBlock, self).__init__()
        self.use_bias = True
        # initialize padding
        if pad_type == 'reflect':
            self.pad = nn.ReflectionPad2d(padding)
        elif pad_type == 'replicate':
            self.pad = nn.ReplicationPad2d(padding)
        elif pad_type == 'zero':
            self.pad = nn.ZeroPad2d(padding)
        elif pad_type == 'none':
            self.pad = None
        else:
            assert 0, "Unsupported padding type: {}".format(pad_type)

        # initialize normalization
        norm_dim = output_dim
        if norm == 'bn':
            self.norm = nn.BatchNorm2d(norm_dim)
        elif norm == 'in':
            self.norm = nn.InstanceNorm2d(norm_dim)
        elif norm == 'none':
            self.norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(norm)

        if weight_norm == 'sn':
            self.weight_norm = spectral_norm_fn
        elif weight_norm == 'wn':
            self.weight_norm = weight_norm_fn
        elif weight_norm == 'none':
            self.weight_norm = None
        else:
            assert 0, "Unsupported normalization: {}".format(weight_norm)

        # initialize activation
        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'elu':
            self.activation = nn.ELU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'none':
            self.activation = None
        else:
            assert 0, "Unsupported activation: {}".format(activation)

        # initialize convolution
        if transpose:
            self.conv = nn.ConvTranspose2d(input_dim, output_dim,
                                           kernel_size, stride,
                                           padding=conv_padding,
                                           output_padding=conv_padding,
                                           dilation=dilation,
                                           bias=self.use_bias)
        else:
            self.conv = nn.Conv2d(input_dim, output_dim, kernel_size, stride,
                                  padding=conv_padding, dilation=dilation,
                                  bias=self.use_bias)

        if self.weight_norm:
            self.conv = self.weight_norm(self.conv)

    def forward(self, x):
        if self.pad:
            x = self.conv(self.pad(x))
        else:
            x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class DCGAN_D(nn.Module):
    def __init__(self, opt):
        super(DCGAN_D, self).__init__()
        self.opt = opt
        model = []

        # start block
        model.append(spectral_norm(nn.Conv2d(self.opt.n_channels, self.opt.D_h_size, kernel_size=4, stride=2, padding=1, bias=False)))
        model.append(nn.LeakyReLU(0.2, inplace=True))

        image_size_new = self.opt.image_size // 2

        # middle block
        mult = 1
        while image_size_new > 4:
            model.append(spectral_norm(nn.Conv2d(self.opt.D_h_size * mult, self.opt.D_h_size * (2*mult), kernel_size=4, stride=2, padding=1, bias=False)))
            model.append(nn.LeakyReLU(0.2, inplace=True))

            image_size_new = image_size_new // 2
            mult *= 2

        self.model = nn.Sequential(*model)
        self.mult = mult

        # end block
        in_size  = int(opt.D_h_size * mult * 4 * 4)
        out_size = self.opt.num_outcomes
        self.fc = spectral_norm(nn.Linear(in_size, out_size, bias=False))

        # resampling trick
        self.reparam = spectral_norm(nn.Linear(in_size, out_size * 2, bias=False))

    def forward(self, input):
        y = self.model(input)

        y = y.view(-1, self.opt.D_h_size * self.mult * 4 * 4)
        output = self.fc(y).view(-1, self.opt.num_outcomes)

        # re-parameterization trick
        if self.opt.use_adaptive_reparam:
            stat_tuple = self.reparam(y).unsqueeze(2).unsqueeze(3)
            mu, logvar = stat_tuple.chunk(2, 1)
            std = logvar.mul(0.5).exp_()
            epsilon = torch.randn(input.shape[0], self.opt.num_outcomes, 1, 1).to(stat_tuple)
            output = epsilon.mul(std).add_(mu).view(-1, self.opt.num_outcomes)

        return output
