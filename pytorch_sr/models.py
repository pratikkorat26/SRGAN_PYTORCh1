import torch
import torch.nn as nn
import torch.nn.functional as F


class ResBlock(nn.Module):
    def __init__(self , in_channels = 64, out_channels = 64 , kernel_size = (3,3) , stride = 1 , activation = "prelu"):
        super(ResBlock, self).__init__()
        if activation == "prelu":
            self.activation = nn.PReLU()
        if activation == "leakyrelu":
            self.activation = nn.LeakyReLU()
        self.conv1 = nn.Conv2d(in_channels = in_channels, out_channels = out_channels,
                               kernel_size = kernel_size, stride = stride , padding = 1)
        self.batchnorm1 = nn.BatchNorm2d(num_features = in_channels)
        self.conv2 = nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                               kernel_size=kernel_size,stride=stride , padding = 1)
        self.batchnorm2 = nn.BatchNorm2d(num_features=in_channels)

    def forward(self , x):
        x1 = self.conv1(x)
        x1 = self.batchnorm1(x1)
        x1 = self.activation(x1)
        x1 = self.conv2(x1)
        x1 = self.batchnorm2(x1)
        x1 = torch.add(x1 , x)

        return x1



class Generator(nn.Module):
    def __init__(self , res_block = 16):
        super(Generator , self).__init__()

        #initial part of network
        self.init_conv = nn.Conv2d(in_channels = 3, out_channels = 64, kernel_size = (3,3), stride = 1 , padding = 1)
        self.init_activation = nn.PReLU()

        # Num Resnet block
        self.num_res = res_block
        self.res_list = []
        for _ in range(self.num_res):
            self.res_list.append(ResBlock())

        self.resnets = nn.Sequential(*self.res_list)

        #End_part of network

        self.conv_end1 = nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3,3), stride=1 , padding = 1)
        self.batchnorm_end1 = nn.BatchNorm2d(num_features=64)
        #ELementwise sum with initial output

        #self end conv 2
        self.conv_end2 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=1 , padding = 1)
        self.pixelshuffle2 = nn.PixelShuffle(upscale_factor = 2)

        self.conv_end3 = nn.Conv2d(in_channels=64, out_channels=256, kernel_size=(3, 3), stride=1 , padding = 1)
        self.pixelshuffle3 = nn.PixelShuffle(upscale_factor=2)

        #final conv for 3 channel
        self.final_conv = nn.Conv2d(in_channels = 64 , out_channels = 3 , kernel_size = 1 , stride = 1 , padding = 0)


    def forward(self , inputs):
        n = self.init_conv(inputs)
        n = self.init_activation(n)

        temp = n

        n = self.resnets(n)

        n = self.conv_end1(n)
        n = self.batchnorm_end1(n)
        n = torch.add(temp , n)

        n = self.conv_end2(n)
        n = self.pixelshuffle2(n)
        n = self.init_activation(n)

        n = self.conv_end3(n)
        n = self.pixelshuffle3(n)

        out = self.final_conv(n)

        return out


class DiscRenBlock(nn.Module):
    def __init__(self , in_channels, out_channel ,kernel_size , stride):
        super(DiscRenBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = out_channel , kernel_size = kernel_size,
                               stride = stride , padding = 1)
        self.batch_norm = nn.BatchNorm2d(num_features = out_channel)
        self.activation = nn.LeakyReLU()

    def forward(self , inputs):
        x = self.conv1(inputs)
        x = self.batch_norm(x)
        x = self.activation(x)

        return x

class Discriminator(nn.Module):
    def __init__(self , in_channels):
        super(Discriminator, self).__init__()
        #inital part
        self.conv1 = nn.Conv2d(in_channels = in_channels , out_channels = 64 , kernel_size = 3 , stride = 2 , padding = 1)
        self.acti1 = nn.LeakyReLU()

        #residuals
        self.res_1 = DiscRenBlock(in_channels = 64 , out_channel = 128 , kernel_size = 3, stride =1)
        self.res_2 = DiscRenBlock(in_channels=128, out_channel=128, kernel_size=3, stride=1)
        self.res_3 = DiscRenBlock(in_channels=128, out_channel=128, kernel_size=3, stride=1)
        self.res_4 = DiscRenBlock(in_channels=128, out_channel=256, kernel_size=4, stride=2)
        self.res_5 = DiscRenBlock(in_channels=256, out_channel=256, kernel_size=4, stride=2)

        #End part
        self.conv2 = nn.Conv2d(in_channels = 256 , out_channels = 256 , kernel_size = (4,4) , stride = 2)
        self.flatten = nn.Flatten()
        self.linear = nn.Linear(in_features = 57600 , out_features = 1000)
        self.acti2 = nn.Sigmoid()
        self.linearend = nn.Linear(in_features = 1000 , out_features = 1)
        #


    def forward(self , inputs):
        x = self.conv1(inputs)
        x = self.acti1(x)

        x = self.res_1(x)
        x = self.res_2(x)
        x = self.res_3(x)
        x = self.res_4(x)
        x = self.res_5(x)

        x = self.conv2(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.linearend(x)
        x = self.acti2(x)

        return x


if __name__ == '__main__':
    pass
