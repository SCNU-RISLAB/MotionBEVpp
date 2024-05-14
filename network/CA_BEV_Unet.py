# -*- coding: utf-8 -*-
# Developed by Jiapeng Xie
# Modified by jxLiang

import torch
import torch.nn as nn
import torch.nn.functional as F
from network.basic_blocks import inconv, outconv, down, up, double_conv, double_conv_circular, single_conv,  single_conv_circular
from network.BaseBlocks import add_down
from dropblock import DropBlock2D


class CA_Unet(nn.Module):

    def __init__(self, moving_n_class,movable_n_class, n_height, residual, dilation=1, group_conv=False, input_batch_norm=False, dropout=0.,
                 circular_padding=False, dropblock=True,PixelShuffle=False,using_UpBlock = False,drop_out=False):
        super(CA_Unet, self).__init__()
        self.moving_n_class = moving_n_class
        self.movable_n_class = movable_n_class if movable_n_class!=None else None
        
        self.n_height = n_height
        '''
        u-net 输入是：  点云BEV输入 B 32 480 360
                        残差输入 B 4 480 360

            输出： B 32*n_class 480 360 
        '''

        self.network = UNet(moving_n_class * n_height,movable_n_class * n_height if movable_n_class!= None else None, 
                            n_height, residual, dilation, group_conv, input_batch_norm, dropout,
                            circular_padding, dropblock,PixelShuffle,using_UpBlock,drop_out)

    def forward(self, x, res):
        moving,movable = self.network(x, res)

        moving = moving.permute(0, 2, 3, 1)  # 维度 0,1,2,3 -> 0,2,3,1  即4, 480, 360, 32
        moving_new_shape = list(moving.size())[:3] + [self.n_height, self.moving_n_class]
        moving = moving.view(moving_new_shape)
        moving = moving.permute(0, 4, 1, 2, 3)

        if self.movable_n_class!=None:
            movable = movable.permute(0, 2, 3, 1)  # 维度 0,1,2,3 -> 0,2,3,1  即4, 480, 360, 32
            movable_new_shape = list(movable.size())[:3] + [self.n_height, self.movable_n_class]
            movable = movable.view(movable_new_shape)
            movable = movable.permute(0, 4, 1, 2, 3)

            
        else: movable = None

        return moving,movable

class UNet(nn.Module):
    '''加了残差连接模块，可做加不加入movable分支和pixelshuffle实验和残差连接模块对比试验
    '''
    def __init__(self, moving_n_class, movable_n_class,n_height, residual, dilation, group_conv, input_batch_norm, dropout, circular_padding,
                 dropblock,PixelShuffle,*args):
        super(UNet, self).__init__()
        print('unet')
        if movable_n_class != None:
            self.movable = True
        else:self.movable = False

        self.inc = inconv(n_height, 32, dilation, input_batch_norm, circular_padding)
        self.down1 = down(32, 64, dilation, group_conv, circular_padding)
        self.down2 = down(64, 128, dilation, group_conv, circular_padding)
        self.down3 = down(128, 256, dilation, group_conv, circular_padding)
        self.down4 = down(256, 256, dilation, group_conv, circular_padding)

        self.res_inc = inconv(residual, 16, dilation, input_batch_norm, circular_padding)
        self.res_down1 = down(16, 32, dilation, group_conv, circular_padding)
        self.res_down2 = down(32, 64, dilation, group_conv, circular_padding)
        self.res_down3 = down(64, 128, dilation, group_conv, circular_padding)
        # self.res_down4 = down(512, 512, dilation, group_conv, circular_padding)
        
        
        self.CGM0 = CAG(channel_a=32, channel_m=16, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM1 = CAG(channel_a=64, channel_m=32*2, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM2 = CAG(channel_a=128, channel_m=64*2+32, circular_padding=circular_padding, group_conv=group_conv)
        self.CGM3 = CAG(channel_a=256, channel_m=128*2+64+32, circular_padding=circular_padding, group_conv=group_conv)
        self.add_down00 = add_down(16,32,2)
        self.add_down01 = add_down(16,32,4)
        self.add_down02 = add_down(16,32,8)
        self.add_down11 = add_down(32,64,2)
        self.add_down12 = add_down(32,64,4)
        self.add_down22 = add_down(64,128,2)


        self.up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
        self.up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)

        self.moving_dropout = nn.Dropout(p=dropout)
        self.moving_outc = outconv(32, moving_n_class)

        if self.movable == True:
            self.movable_up3 = up(512, 128, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.movable_up2 = up(256, 64, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.movable_up1 = up(128, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.movable_up0 = up(64, 32, circular_padding, group_conv=group_conv, use_dropblock=dropblock, drop_p=dropout,PixelShuffle=PixelShuffle)
            self.movable_dropout = nn.Dropout(p=dropout)
            self.movable_outc = outconv(32,movable_n_class)# 2*n_height(32)

    def forward(self, x, res):
        # x : B 32 480 360
        # res : B 8 480 360
        c0_motion = self.res_inc(res) # B 8 480 360 -> B 16 480 360
        c1_motion = self.res_down1(c0_motion) # B 16 480 360 -> B 32 240 180
        c2_motion = self.res_down2(c1_motion) # B 32 240 180 -> B 64 120 90
        c3_motion = self.res_down3(c2_motion) # B 64 120 90 -> B 128 60 45
        #c4_motion = self.res_down4(c3_motion)

        c0_appearance = self.inc(x)  # B 32 480 360 -> B 32 480 360
        c0_appearance_fused, _ = self.CGM0(c0_appearance, c0_motion) # B 32 480 360 | B 16 480 360 -> B 32 480 360 | B 16 480 360

        c1_appearance = self.down1(c0_appearance_fused)  # B 32 480 360 -> B 64 240 180
        c1_motion_add= torch.cat([c1_motion,self.add_down00(c0_motion)],dim=1) # -> B 64 240 180
        c1_appearance_fused,_ = self.CGM1(c1_appearance,c1_motion_add) # B 64 240 180 | B 64 240 180 -> B 64 240 180 | B 64 240 180

        c2_appearance = self.down2(c1_appearance_fused)  # B 64 240 180 -> B 128 120 90
        c2_motion_add= torch.cat([c2_motion,self.add_down11(c1_motion),self.add_down01(c0_motion)],dim=1) # -> B 160 120 90
        c2_appearance_fused,_ = self.CGM2(c2_appearance,c2_motion_add) # B 128 120 90 | B 160 120 90 -> B 128 120 90 | B 160 120 90

        c3_appearance = self.down3(c2_appearance_fused)  # B 128 120 90 -> B 256 60 45
        c3_motion_add= torch.cat([c3_motion,self.add_down22(c2_motion),self.add_down12(c1_motion),self.add_down02(c0_motion)],dim=1) # -> B 352 60 45
        c3_appearance_fused,_ = self.CGM3(c3_appearance,c3_motion_add) # B 256 60 45 | B 352 60 45 -> B 256 60 45 | B 352 60 45

        c4_appearance = self.down4(c3_appearance_fused)  # B 256 60 45 -> B 256 30 22


        output = self.up3(c4_appearance, c3_appearance_fused) # B 256 30 22 | B 256 60 45 -> B 64 120 90
        output = self.up2(output, c2_appearance_fused) # B 64 120 90 | B 128 120 90 -> B 32 240 180
        output = self.up1(output, c1_appearance_fused)
        output = self.up0(output, c0_appearance_fused)
        moving_output = self.moving_outc(self.moving_dropout(output))

        if self.movable == True:
            movable_out = self.movable_up3(c4_appearance,c3_appearance)
            movable_out = self.movable_up2(movable_out,c2_appearance)
            movable_out = self.movable_up1(movable_out,c1_appearance)
            movable_out = self.movable_up0(movable_out,c0_appearance)
            movable_output = self.movable_outc(self.movable_dropout(movable_out))
        else:movable_output = None
        return moving_output,movable_output

class ChannelAttention(nn.Module):
    def __init__(self, in_planes):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 8, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 8, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        x_channel = x.mul(out)  # out * x
        return x_channel


class SpatialAttention(nn.Module):
    def __init__(self, circular_padding):
        super(SpatialAttention, self).__init__()
        self.circular_padding = circular_padding
        if circular_padding:
            padding = (1, 0)
        else:
            padding = 1

        self.conv1 = nn.Conv2d(2, 1, 3, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        if self.circular_padding:
            out = F.pad(out, (1, 1, 0, 0), mode='circular')
        out = self.conv1(out)
        x_spatial = self.sigmoid(out)
        return x_spatial


class attention_module_MGA_tmc(nn.Module):
    def __init__(self, channel_a, channel_m):
        super(attention_module_MGA_tmc, self).__init__()
        self.conv1x1_channel_wise = nn.Conv2d(channel_a, channel_a, 1, bias=True)
        self.conv1x1_spatial = nn.Conv2d(channel_m, 1, 1, bias=True)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))

    def forward(self, img_feat, flow_feat):
        """
            flow_feat_map:  [bsize, 1, h, w]
            feat_vec:       [bsize, channel, 1, 1]
            channel_attentioned_img_feat:  [bsize, channel, h, w]
        """
        # spatial attention
        flow_feat_map = self.conv1x1_spatial(flow_feat)
        flow_feat_map = nn.Sigmoid()(flow_feat_map)
        spatial_attentioned_img_feat = flow_feat_map * img_feat

        # channel-wise attention
        feat_vec = self.avg_pool(spatial_attentioned_img_feat)
        feat_vec = self.conv1x1_channel_wise(feat_vec)
        feat_vec = nn.Softmax(dim=1)(feat_vec) * feat_vec.shape[1]
        channel_attentioned_img_feat = spatial_attentioned_img_feat * feat_vec

        final_feat = channel_attentioned_img_feat + img_feat
        return final_feat


class CAG(nn.Module):  # Co-Attention Gate Module
    def __init__(self, channel_a, channel_m, circular_padding=False, group_conv=False):
        super(CAG, self).__init__()
        self.circular_padding = circular_padding
        self.channel_a = channel_a
        self.channel_m = channel_m

        if circular_padding:
            self.fuse_feature1 = single_conv_circular(channel_a + channel_m, 32, group_conv)
            self.fuse_feature2 = nn.Conv2d(32, 2, kernel_size=3, padding=(1, 0))
        else:
            self.fuse_feature1 = single_conv(channel_a + channel_m, 32, group_conv)
            self.fuse_feature2 = nn.Conv2d(32, 2, kernel_size=3, padding=1)
        self.motion_channel_attention = ChannelAttention(channel_m)
        self.motion_spatial_attention = SpatialAttention(circular_padding)
        self.attention_module = attention_module_MGA_tmc(channel_a, channel_m)

    def forward(self, f_a, f_m):
        g_am = self.fuse_feature1(torch.cat((f_a, f_m), dim=1))
        if self.circular_padding:
            g_am = F.pad(g_am, (1, 1, 0, 0), mode='circular')
        g_am = self.fuse_feature2(g_am)
        g_am = F.adaptive_avg_pool2d(torch.sigmoid(g_am), 1)
        g_a = g_am[:, 0, :, :].unsqueeze(1).repeat(1, self.channel_a, 1, 1) * f_a
        g_m = g_am[:, 1, :, :].unsqueeze(1).repeat(1, self.channel_m, 1, 1) * f_m

        e_am = self.attention_module(g_a, g_m)

        return e_am, g_m


class MCM(nn.Module):  # Motion Correction Module
    def __init__(self, channel_a_in, channel_a_out, channel_m_in, channel_m_out, channel_out,
                 compute_sim=False, circular_padding=False, group_conv=False):
        super(MCM, self).__init__()

        self.attention_module = attention_module_MGA_tmc(channel_a_out, channel_m_out)

        if circular_padding:
            self.gate_a = single_conv_circular(channel_a_in, channel_a_out, group_conv)
            self.gate_m = single_conv_circular(channel_m_in, channel_m_out, group_conv)
            self.fuse_appearance = double_conv_circular(channel_a_out * 2, channel_a_out, group_conv)
            self.fuse_motion = double_conv_circular(channel_m_out * 2, channel_m_out, group_conv)
            self.out_conv = single_conv_circular(channel_a_out, channel_out, group_conv)
        else:
            self.gate_a = single_conv(channel_a_in, channel_a_out, group_conv)
            self.gate_m = single_conv(channel_m_in, channel_m_out, group_conv)
            self.fuse_appearance = double_conv(channel_a_out * 2, channel_a_out, group_conv)
            self.fuse_motion = double_conv(channel_m_out * 2, channel_m_out, group_conv)
            self.out_conv = single_conv(channel_a_out, channel_out, group_conv)

        self.dropblock = DropBlock2D(block_size=7, drop_prob=0.5)
        self.compute_sim = compute_sim

    def forward(self, e_am, g_m, d_last):
        gated_a = self.gate_a(e_am)
        gated_m = self.gate_m(g_m)

        if self.compute_sim:
            batch, channel, h, w = gated_a.shape
            M = h * w
            appearance_features = gated_a.view(batch, channel, M)
            motion_features = gated_m.view(batch, channel, M).permute(0, 2, 1)
            p_4 = torch.matmul(motion_features, appearance_features)
            p_4 = F.softmax((channel ** -.5) * p_4, dim=-1)
            d_last_up = torch.matmul(p_4, appearance_features.permute(0, 2, 1)).permute(0, 2, 1).view(batch, channel, h,
                                                                                                      w)
        else:
            d_last_up = F.interpolate(d_last, size=gated_a.size()[2:], mode='bilinear', align_corners=True)

        f_motion = self.fuse_motion(torch.cat((gated_m, d_last_up), dim=1))
        f_appearance = self.fuse_appearance(torch.cat((gated_a, d_last_up), dim=1))
        f_attention = self.attention_module(f_appearance, f_motion)
        d_out = self.out_conv(f_attention + d_last_up)

        d_out = self.dropblock(d_out)
        return d_out

