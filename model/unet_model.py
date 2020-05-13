from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, n_channels, n_classes, bilinear=True):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear

        n_filter = [64, 128, 256, 512, 1024]  # [32, 64, 128, 256, 512]  #

        self.inc = DoubleConv(n_channels, n_filter[0])
        self.down1 = Down(n_filter[0], n_filter[1])
        self.down2 = Down(n_filter[1], n_filter[2])
        self.down3 = Down(n_filter[2], n_filter[3])
        factor = 2 if bilinear else 1
        self.down4 = Down(n_filter[3], n_filter[4] // factor)
        self.up1 = Up(n_filter[4], n_filter[3] // factor, bilinear)
        self.up2 = Up(n_filter[3], n_filter[2] // factor, bilinear)
        self.up3 = Up(n_filter[2], n_filter[1] // factor, bilinear)
        self.up4 = Up(n_filter[1], n_filter[0], bilinear)
        self.outc = OutConv(n_filter[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        out = self.outc(x)
        return out
