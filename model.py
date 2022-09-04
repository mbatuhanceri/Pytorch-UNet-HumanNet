import torch
from torch import nn



class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.sequence = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU())
    
    def forward(self, x):
        return self.sequence(x)


class InConv(nn.Module):
    def __init__(self, out_channels):
        super(InConv, self).__init__()
        self.input_conv = DoubleConv(1, out_channels)

    def forward(self, x):
        return self.input_conv(x)


class DownConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownConv, self).__init__()
        self.sequence = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels))

    def forward(self, x):
        return self.sequence(x)
    

class UpConv(nn.Module):
    def __init__(self, in_channels, skip_channels ,out_channels):
        super(UpConv, self).__init__()
        self.up_sample = nn.Upsample(scale_factor=2, mode='nearest')
        self.double_conv = DoubleConv(in_channels + skip_channels, out_channels)

    def forward(self, x, x_skip):
        x = self.up_sample(x)
        x = torch.cat((x, x_skip), dim=1)
        x = self.double_conv(x)
        return x

class UNet(nn.Module):
    def __init__(self):
        super(UNet ,self).__init__()

        self.in_conv = InConv(16)
        self.down_1 = DownConv(16, 32)
        self.down_2 = DownConv(32, 64)
        self.down_3 = DownConv(64, 128)
        self.down_4 = DownConv(128, 128)
        
        self.up_4 = UpConv(128, 128, 128)
        self.up_3 = UpConv(128, 64, 64)
        self.up_2 = UpConv(64, 32, 32)
        self.up_1 = UpConv(32, 16, 16)
        self.out_conv = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=5, stride=1, padding=2)

    def forward(self, input):
        x0 = self.in_conv(input)
        x1 = self.down_1(x0)
        x = self.down_2(x1)
        # x3 = self.down_3(x2)
        # x = self.down_4(x3)
        # x = self.up_4(x, x3)
        # x = self.up_3(x, x2)
        x = self.up_2(x, x1)
        x = self.up_1(x, x0)
        x = self.out_conv(x)

        return(x)