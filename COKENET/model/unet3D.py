from torch.nn import Module, Sequential
from torch.nn import Conv3d, ConvTranspose3d, BatchNorm3d, MaxPool3d, AvgPool1d
from torch.nn import ReLU, Sigmoid
import torch
import torch.nn.functional as F

class UNet3D(Module):
    # __                            __
    #  1|__   ________________   __|1
    #     2|__  ____________  __|2
    #        3|__  ______  __|3
    #           4|__ __ __|4
    # The convolution operations on either side are residual subject to 1*1 Convolution for channel homogeneity

    def __init__(self, num_channels=24, feat_channels=[24,24], residual='conv'):
        # residual: conv for residual input x through 1*1 conv across every layer for downsampling, None for removal of residuals

        super(UNet3D, self).__init__()

        # Encoder downsamplers
        self.pool1 = MaxPool3d((1, 2, 2))
        self.pool2 = MaxPool3d((1, 2, 2))
        self.pool3 = MaxPool3d((1, 2, 2))
        self.pool4 = MaxPool3d((1, 2, 2))

        # Encoder convolutions
        self.conv_blk1 = Conv3D_Block(num_channels, feat_channels[0], residual=residual)
        self.conv_blk2 = Conv3D_Block(feat_channels[0], feat_channels[1], residual=residual)
        # self.conv_blk3 = Conv3D_Block(feat_channels[1], feat_channels[2], residual=residual)
        # self.conv_blk4 = Conv3D_Block(feat_channels[2], feat_channels[3], residual=residual)
        # self.conv_blk5 = Conv3D_Block(feat_channels[3], feat_channels[4], residual=residual)

        # Decoder convolutions
        # self.dec_conv_blk4 = Conv3D_Block(2 * feat_channels[3], feat_channels[3], residual=residual)
        # self.dec_conv_blk3 = Conv3D_Block(2 * feat_channels[2], feat_channels[2], residual=residual)
        self.dec_conv_blk2 = Conv3D_Block(2 * feat_channels[1], feat_channels[1], residual=residual)
        self.dec_conv_blk1 = Conv3D_Block(2 * feat_channels[0], feat_channels[0], residual=residual)

        # Decoder upsamplers
        # self.deconv_blk4 = Deconv3D_Block(feat_channels[4], feat_channels[3])
        # self.deconv_blk3 = Deconv3D_Block(feat_channels[3], feat_channels[2])
        # self.deconv_blk2 = Deconv3D_Block(feat_channels[2], feat_channels[1])
        self.deconv_blk1 = Deconv3D_Block(feat_channels[1], feat_channels[0])

        # Final 1*1 Conv Segmentation map
        self.one_conv = Conv3d(feat_channels[0], num_channels, kernel_size=1, stride=1, padding=0, bias=True)  #24
        # self.one_conv = Conv3d(feat_channels[0], 1, kernel_size=1, stride=1, padding=0, bias=True)
        # Activation function
        self.sigmoid = Sigmoid()


    def forward(self, x):
        # Encoder part

        x1 = self.conv_blk1(x)

        x_low1 = self.pool1(x1)
        base = self.conv_blk2(x_low1)

        # base = self.pool2(x2)
        # # x3 = self.conv_blk3(x_low2)
        # base = self.conv_blk3(x_low2)

        # x_low3 = self.pool3(x3)
        # x4 = self.conv_blk4(x_low3)
        #
        # x_low4 = self.pool4(x4)
        # base = self.conv_blk5(x_low4)

        # Decoder part

        # d4 = torch.cat([self.deconv_blk4(base), x4], dim=1)
        # d_high4 = self.dec_conv_blk4(d4)

        # d3 = torch.cat([self.deconv_blk3(d_high4), x3], dim=1)
        # d_high3 = self.dec_conv_blk3(d3)

        # d2 = torch.cat([self.deconv_blk2(base), x2], dim=1)
        # d_high2 = self.dec_conv_blk2(d2)
        try:
            d1 = torch.cat([self.deconv_blk1(base), x1], dim=1)
        except:
            apool = torch.nn.AdaptiveAvgPool3d((2, x1.size(3), x1.size(4)))
            base = apool(self.deconv_blk1(base))
            d1 = torch.cat([base, x1], dim=1)

        d_high1 = self.dec_conv_blk1(d1)

        APTpool = torch.nn.AdaptiveAvgPool3d((1,d_high1.size(3),d_high1.size(4)))
        # seg = self.sigmoid(self.one_conv(d_high1))

        output = self.one_conv(d_high1)
        output_pool = APTpool(output)
        output_final = output_pool.squeeze(2)
        # output_final  = torch.ones_like(output_final).cuda() + self.sigmoid(output_final) # 24
        # output_final = 2 *self.sigmoid(output_final)
        # output_final = F.relu(output_final)

        output_final = torch.ones_like(output_final).cuda() + self.sigmoid(output_final)

        return output_final



class Conv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=3, stride=1, padding=1, residual=None):

        super(Conv3D_Block, self).__init__()

        self.conv1 = Sequential(
            Conv3d(inp_feat, out_feat, kernel_size=kernel,
                   stride=stride, padding=padding, bias=True),
            BatchNorm3d(out_feat),
            ReLU())

        self.conv2 = Sequential(
            Conv3d(out_feat, out_feat, kernel_size=kernel,
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


class Deconv3D_Block(Module):

    def __init__(self, inp_feat, out_feat, kernel=4, stride=2, padding=1):
        super(Deconv3D_Block, self).__init__()

        self.deconv = Sequential(
            ConvTranspose3d(inp_feat, out_feat, kernel_size=(1, kernel, kernel),
                            stride=(1, stride, stride), padding=(0, padding, padding), output_padding=0, bias=True),
            ReLU())

    def forward(self, x):
        return self.deconv(x)


class ChannelPool3d(AvgPool1d):

    def __init__(self, kernel_size, stride, padding):
        super(ChannelPool3d, self).__init__(kernel_size, stride, padding)
        self.pool_1d = AvgPool1d(self.kernel_size, self.stride, self.padding, self.ceil_mode)

    def forward(self, inp):
        n, c, d, w, h = inp.size()
        inp = inp.view(n, c, d * w * h).permute(0, 2, 1)
        pooled = self.pool_1d(inp)
        c = int(c / self.kernel_size[0])
        return inp.view(n, c, d, w, h)


if __name__ == '__main__':
    net = UNet3D(residual='pool')

    import torch

    x = torch.ones(1, 32, 1, 128, 128)

    print(net.forward(x).size())