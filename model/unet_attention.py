from .unet_parts import *
import pdb


class UNetAttention(nn.Module):
    def __init__(self, in_channels, n_classes):
        super(UNetAttention, self).__init__()
        n_filter = [64, 128, 256, 512, 1024]  # [32, 64, 128, 256, 512]  #

        self.inc = DoubleConv(in_channels, n_filter[0])
        self.down1 = Down(n_filter[0], n_filter[1])
        self.down2 = Down(n_filter[1], n_filter[2])
        self.down3 = Down(n_filter[2], n_filter[3])
        self.down4 = Down(n_filter[3], n_filter[4])

        self.upconv1 = UpConv(n_filter[4], n_filter[3])
        self.att1 = AttentionBlock(f_g=n_filter[3], f_l=n_filter[3], f_int=n_filter[2])
        self.double_conv1 = DoubleConv(n_filter[4], n_filter[3])

        self.upconv2 = UpConv(n_filter[3], n_filter[2])
        self.att2 = AttentionBlock(f_g=n_filter[2], f_l=n_filter[2], f_int=n_filter[1])
        self.double_conv2 = DoubleConv(n_filter[3], n_filter[2])

        self.upconv3 = UpConv(n_filter[2], n_filter[1])
        self.att3 = AttentionBlock(f_g=n_filter[1], f_l=n_filter[1], f_int=n_filter[0])
        self.double_conv3 = DoubleConv(n_filter[2], n_filter[1])

        self.upconv4 = UpConv(n_filter[1], n_filter[0])
        self.att4 = AttentionBlock(f_g=n_filter[0], f_l=n_filter[0], f_int=int(n_filter[0] / 2))
        self.double_conv4 = DoubleConv(n_filter[1], n_filter[0])

        self.outc = OutConv(n_filter[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        d5 = self.upconv1(x5, pad_like=x4)
        x4 = self.att1(g=d5, x=x4)
        d5 = torch.cat((x4, d5), dim=1)
        d5 = self.double_conv1(d5)

        d4 = self.upconv2(d5, pad_like=x3)
        x3 = self.att2(g=d4, x=x3)
        d4 = torch.cat((x3, d4), dim=1)
        d4 = self.double_conv2(d4)

        d3 = self.upconv3(d4, pad_like=x2)
        x2 = self.att3(g=d3, x=x2)
        d3 = torch.cat((x2, d3), dim=1)
        d3 = self.double_conv3(d3)

        d2 = self.upconv4(d3, pad_like=x1)
        x1 = self.att4(g=d2, x=x1)
        d2 = torch.cat((x1, d2), dim=1)
        x = self.double_conv4(d2)

        return self.outc(x)
