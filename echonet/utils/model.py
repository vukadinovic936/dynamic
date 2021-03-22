import math
from torch.nn import Module, Sequential, Parameter
from torch.nn import Conv2d, Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d, Upsample, Linear, Dropout2d
from torch.nn import ReLU, Sigmoid, Tanh
import torch
from torch.autograd import Variable
import torch.nn.functional as F
import pdb

class unet3d(Module):
    def __init__(self, num_channels=3, feat_channels=[32, 64, 128, 256, 256], residual='conv', is_dilated=False, is_off_unit=False, is_2d1d=False):
        super(unet3d, self).__init__()
        
        self.is_off_unit = is_off_unit

        # Encoder downsamplers
        self.pool1 = MaxPool3d((1,2,2))
        self.pool2 = MaxPool3d((1,2,2))
        self.pool3 = MaxPool3d((1,2,2))
        self.pool4 = MaxPool3d((1,2,2))

        # Encoder convolutions
        if is_dilated:
            self.conv_blk1 = Conv3D_Block_adv(num_channels, feat_channels[0], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk2 = Conv3D_Block_adv(feat_channels[0], feat_channels[1], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk3 = Conv3D_Block_adv(feat_channels[1], feat_channels[2], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk4 = Conv3D_Block_adv(feat_channels[2], feat_channels[3], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk5 = Conv3D_Block_adv(feat_channels[3], feat_channels[4], residual=residual, is_2d1d=is_2d1d)

            # Decoder convolutions
            self.dec_conv_blk4 = Conv3D_Block_adv(2*feat_channels[3], feat_channels[3], residual=residual, is_2d1d=is_2d1d)
            self.dec_conv_blk3 = Conv3D_Block_adv(2*feat_channels[2], feat_channels[2], residual=residual, is_2d1d=is_2d1d)
            self.dec_conv_blk2 = Conv3D_Block_adv(2*feat_channels[1], feat_channels[1], residual=residual, is_2d1d=is_2d1d)
            self.dec_conv_blk1 = Conv3D_Block_adv(2*feat_channels[0], feat_channels[0], residual=residual, is_2d1d=is_2d1d)

        else:
            # Encoder convolutions
            self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual, is_2d1d=is_2d1d)
            self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual, is_2d1d=is_2d1d)

            # Decoder convolutions
            self.dec_conv_blk4 = Conv3D_Block(2*feat_channels[3], feat_channels[3], residual=residual, is_2d1d=is_2d1d)
            self.dec_conv_blk3 = Conv3D_Block(2*feat_channels[2], feat_channels[2], residual=residual, is_2d1d=is_2d1d)
            self.dec_conv_blk2 = Conv3D_Block(2*feat_channels[1], feat_channels[1], residual=residual, is_2d1d=is_2d1d)
            self.dec_conv_blk1 = Conv3D_Block(2*feat_channels[0], feat_channels[0], residual=residual, is_2d1d=is_2d1d)

        if self.is_off_unit:
            self.off_unit_enc1 = OFF_Unit(feat_channels[0], feat_channels[0])
            self.off_unit_enc2 = OFF_Unit(feat_channels[1], feat_channels[1])
            self.off_unit_enc3 = OFF_Unit(feat_channels[2], feat_channels[2])
            self.off_unit_enc4 = OFF_Unit(feat_channels[3], feat_channels[3])
            self.off_unit_enc5 = OFF_Unit(feat_channels[4], feat_channels[4])

            self.off_unit_dec4 = OFF_Unit(feat_channels[3], feat_channels[3])
            self.off_unit_dec3 = OFF_Unit(feat_channels[2], feat_channels[2])
            self.off_unit_dec2 = OFF_Unit(feat_channels[1], feat_channels[1])
            self.off_unit_dec1 = OFF_Unit(feat_channels[0], feat_channels[0])

        # Decoder upsamplers
        self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        if self.is_off_unit:
            # Encoder part
            #pdb.set_trace()

            x1 = self.conv_blk1(x)
            x1 = self.off_unit_enc1(x1)

            x_low1 = self.pool1(x1)
            x2 = self.conv_blk2(x_low1)
            x2 = self.off_unit_enc2(x2)

            x_low2 = self.pool2(x2)
            x3 = self.conv_blk3(x_low2)
            x3 = self.off_unit_enc3(x3)

            x_low3 = self.pool3(x3)
            x4 = self.conv_blk4(x_low3)
            x4 = self.off_unit_enc4(x4)

            x_low4 = self.pool4(x4)
            base = self.conv_blk5(x_low4)
            base = self.off_unit_enc5(base)

            # Decoder part
            d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
            d_high4 = self.dec_conv_blk4(d4)
            d_high4 = self.off_unit_dec4(d_high4)

            d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
            d_high3 = self.dec_conv_blk3(d3)
            d_high3 = self.off_unit_dec3(d_high3)

            d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
            d_high2 = self.dec_conv_blk2(d2)
            d_high2 = self.off_unit_dec2(d_high2)

            d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
            d_high1 = self.dec_conv_blk1(d1)
            d_high1 = self.off_unit_dec1(d_high1)
        else:
            # Encoder part
            #pdb.set_trace()

            x1 = self.conv_blk1(x)

            x_low1 = self.pool1(x1)
            x2 = self.conv_blk2(x_low1)

            x_low2 = self.pool2(x2)
            x3 = self.conv_blk3(x_low2)

            x_low3 = self.pool3(x3)
            x4 = self.conv_blk4(x_low3)

            x_low4 = self.pool4(x4)
            base = self.conv_blk5(x_low4)

            # Decoder part

            d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
            d_high4 = self.dec_conv_blk4(d4)

            d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
            d_high3 = self.dec_conv_blk3(d3)

            d2 = torch.cat([self.deconv_blk2(d_high3), x2], dim=1)
            d_high2 = self.dec_conv_blk2(d2)

            d1 = torch.cat([self.deconv_blk1(d_high2), x1], dim=1)
            d_high1 = self.dec_conv_blk1(d1)

        seg = self.one_conv(d_high1)

        return seg

class Conv3D_Block(Module):
    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None, is_2d1d=False):
        super(Conv3D_Block, self).__init__()
        if is_2d1d:
            conv_type = Conv2d1d
        else:
            conv_type = Conv3d
        self.conv1 = Sequential(
                        conv_type(inp_feat, out_feat, kernel_size=kernel,
                                stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.conv2 = Sequential(
                        conv_type(out_feat, out_feat, kernel_size=kernel,
                                stride=stride, padding=padding, bias=True),
                        BatchNorm3d(out_feat),
                        ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(inp_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        res = x
        if not self.residual:
            return self.conv2(self.conv1(x))
        else:
            return self.conv2(self.conv1(x)) + self.residual_upsampler(res)

class Conv3D_Block_adv(Module):

    def __init__(self, inp_feat, out_feat, residual=None, is_2d1d=False):
        super(Conv3D_Block_adv, self).__init__()

        if is_2d1d:
            conv_type = Conv2d1d
        else:
            conv_type = Conv3d
        self.conv1 = conv_type(inp_feat, out_feat, kernel_size=1, dilation=1,
                                    stride=1, padding=0, bias=True)
        self.conv2_1 = conv_type(out_feat, out_feat, kernel_size=3, dilation=1,
                                    stride=1, padding=1, bias=True)
        self.conv2_2 = conv_type(out_feat, out_feat, kernel_size=3, dilation=3,
                                    stride=1, padding=3, bias=True)
        #self.conv2_3 = Conv3d(out_feat, out_feat, kernel_size=3, dilation=5,
        #                            stride=1, padding=5, bias=True)
        self.conv2 = Sequential(
                        BatchNorm3d(out_feat),
                        ReLU())

        self.residual = residual

        if self.residual is not None:
            self.residual_upsampler = Conv3d(out_feat, out_feat, kernel_size=1, bias=False)

    def forward(self, x):
        #pdb.set_trace()
        conv1 = self.conv1(x)
        conv2_1 = self.conv2_1(conv1)
        conv2_2 = self.conv2_2(conv1)
        #conv2_3 = self.conv2_3(conv1)
        #conv2_add = conv2_1 + conv2_2 + conv2_3
        conv2_add = conv2_1 + conv2_2
        if self.residual != None:
            conv2_add = conv2_add + self.residual_upsampler(conv1)
        conv2 = self.conv2(conv2_add)
        return conv2

class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):

        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
                        ConvTranspose3d(inp_feat, out_feat, kernel_size=(1,kernel,kernel),
                                    stride=(1,stride,stride), padding=(0, padding, padding), output_padding=0, bias=True),
                        ReLU())

    def forward(self, x):

        return self.deconv(x)

class Conv2d1d(Module):
    def __init__(self, inp_feat, out_feat, kernel_size=3, stride=1, padding=1, bias=True):
        super(Conv2d1d, self).__init__()
        mid_feat = int(kernel_size*kernel_size*inp_feat*out_feat/(out_feat+kernel_size*inp_feat))
        self.conv_spatial = Sequential(
                               Conv3d(inp_feat, mid_feat, kernel_size=(1,kernel_size,kernel_size),
                                       stride=(1,stride,stride), padding=(0,padding,padding), bias=bias),
                               BatchNorm3d(mid_feat),
                               ReLU())
        self.conv_temporal = Conv3d(mid_feat, out_feat, kernel_size=(kernel_size,1,1),
                                    stride=(stride,1,1), padding=(padding,0,0), bias=bias)

    def forward(self, x):
        conv_s = self.conv_spatial(x)
        conv_t = self.conv_temporal(conv_s)
        return conv_t

class OFF_Unit(Module):
    def __init__(self, inp_feat, out_feat):
        super(OFF_Unit, self).__init__()
        mid_feat = inp_feat // 2
        self.sobel_x = torch.tensor([[-1., 0., 1.],
                                     [-1., 0., 1.],
                                     [-1., 0., 1.]]).view(1,1,3,3)
        self.sobel_y = torch.tensor([[-1., -1., -1.],
                                     [0., 0., 0.],
                                     [1., 1., 1.]]).view(1,1,3,3)
        if torch.cuda.is_available():
            device = torch.device("cuda")
            self.sobel_x = self.sobel_x.to(device)
            self.sobel_y = self.sobel_y.to(device)
        self.conv1_temporal = Conv3d(inp_feat, mid_feat, kernel_size=1, bias=True)
        self.conv1_spatial = Conv3d(inp_feat, mid_feat, kernel_size=1, bias=True)
        self.conv2 = Conv3d(3*mid_feat+inp_feat, out_feat, kernel_size=1, bias=True)

    def forward(self, x):
        # temporal
        x_temporal = self.conv1_temporal(x)
        feat_temporal = x_temporal[:,:,1:] - x_temporal[:,:,:-1]
        feat_temporal = torch.cat([feat_temporal, feat_temporal[:,:,-2:-1]], dim=2)
        # spatial
        x_spatial = self.conv1_spatial(x)
        x_reshape = torch.transpose(x_spatial, 1, 2).reshape(-1, x_spatial.shape[1], x.shape[3], x.shape[4]) # (bs*ts, ch, w, h)
        feat_x = []
        feat_y = []
        for ch_i in range(x_reshape.shape[1]):
            feat_xi = F.conv2d(x_reshape[:,ch_i,...].unsqueeze(1), self.sobel_x, padding=1)
            feat_yi = F.conv2d(x_reshape[:,ch_i,...].unsqueeze(1), self.sobel_y, padding=1)
            feat_x.append(feat_xi)
            feat_y.append(feat_yi)
        feat_x = torch.cat(feat_x, dim=1)
        feat_y = torch.cat(feat_y, dim=1)
        feat_x = torch.transpose(feat_x.reshape(x.shape[0], x.shape[2], x_spatial.shape[1], x.shape[3], x.shape[4]), 1, 2)
        feat_y = torch.transpose(feat_y.reshape(x.shape[0], x.shape[2], x_spatial.shape[1], x.shape[3], x.shape[4]), 1, 2)
        # concat
        feat = torch.cat([feat_x, feat_y, feat_temporal, x], dim=1)
        output = self.conv2(feat)
        return output
