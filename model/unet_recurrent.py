from .unet_parts import *
import pdb

class RecurrentUNet(nn.Module):
    def __init__(self, in_channels, n_classes, t=2):
        super(RecurrentUNet, self).__init__()

        n_filter = [64, 128, 256, 512, 1024]  # [32, 64, 128, 256, 512]  #

        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.rec_down1 = RecurrentConv(in_channels, n_filter[0], t)
        self.rec_down2 = RecurrentConv(n_filter[0], n_filter[1], t)
        self.rec_down3 = RecurrentConv(n_filter[1], n_filter[2], t)
        self.rec_down4 = RecurrentConv(n_filter[2], n_filter[3], t)
        self.rec_down5 = RecurrentConv(n_filter[3], n_filter[4], t)

        self.up5 = UpConv(n_filter[4], n_filter[3])
        self.rec_up5 = RecurrentConv(n_filter[4], n_filter[3], t)

        self.up4 = UpConv(n_filter[3], n_filter[2])
        self.rec_up4 = RecurrentConv(n_filter[3], n_filter[2], t)

        self.up3 = UpConv(n_filter[2], n_filter[1])
        self.rec_up3 = RecurrentConv(n_filter[2], n_filter[1], t)

        self.up2 = UpConv(n_filter[1], n_filter[0])
        self.rec_up2 = RecurrentConv(n_filter[1], n_filter[0], t)

        self.outconv = nn.Conv2d(64, n_classes, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        x1 = self.rec_down1(x)

        x2 = self.maxpool(x1)
        x2 = self.rec_down2(x2)

        x3 = self.maxpool(x2)
        x3 = self.rec_down3(x3)

        x4 = self.maxpool(x3)
        x4 = self.rec_down4(x4)

        x5 = self.maxpool(x4)
        x5 = self.rec_down5(x5)

        up5 = self.up5(x5, pad_like=x4)
        up5 = torch.cat((x4, up5), dim=1)
        up5 = self.rec_up5(up5)

        up4 = self.up4(up5, pad_like=x3)
        up4 = torch.cat((x3, up4), dim=1)
        up4 = self.rec_up4(up4)

        up3 = self.up3(up4, pad_like=x2)
        up3 = torch.cat((x2, up3), dim=1)
        up3 = self.rec_up3(up3)

        up2 = self.up2(up3, pad_like=x1)
        up2 = torch.cat((x1, up2), dim=1)
        up2 = self.rec_up2(up2)

        out = self.outconv(up2)
        return out
