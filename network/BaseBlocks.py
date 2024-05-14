#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.


import torch
import torch.nn as nn
import torch.nn.functional as F


class cat_block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(cat_block, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
        )

        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=out_channel,
                out_channels=out_channel,
                kernel_size=3,
                stride=1,
                padding=1,
            ),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channel),
        )

    def forward(self, x):
        out = self.conv2(self.conv1(x))
        return out
    
class cat_down(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()

        self.down = nn.Sequential(
            nn.MaxPool2d(kernel_size=kernel_size),
        )

    def forward(self, x):
        out = self.down(x)
        return out    
class add_down(nn.Module):
    def __init__(self,inc,outc,kernel_size):
        super(add_down,self).__init__()

        self.down = cat_down(kernel_size=kernel_size)
        self.conv = cat_block(inc,outc)


    def forward(self,x):
        out = self.conv(self.down(x))
        # print("out",out.shape)
        return out
