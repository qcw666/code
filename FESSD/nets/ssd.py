import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from myssd.nets.mobilenetv2 import InvertedResidual, mobilenet_v2
from myssd.nets.vgg import vgg as add_vgg

from myssd.nets.ECANET import eca_block

class L2Norm(nn.Module):
    def __init__(self, n_channels, scale):
        super(L2Norm, self).__init__()
        self.n_channels = n_channels
        self.gamma = scale or None
        self.eps = 1e-10
        self.weight = nn.Parameter(torch.Tensor(self.n_channels))
        self.reset_parameters()

    def reset_parameters(self):
        init.constant_(self.weight, self.gamma)

    def forward(self, x):
        norm = x.pow(2).sum(dim=1, keepdim=True).sqrt() + self.eps
        # x /= norm
        x = torch.div(x, norm)
        out = self.weight.unsqueeze(0).unsqueeze(2).unsqueeze(3).expand_as(x) * x
        return out


def add_extras(in_channels, backbone_name):
    layers = []
    if backbone_name == 'vgg':
        # Block 6
        # 19,19,1024 -> 19,19,256 -> 10,10,512
        layers += [nn.Conv2d(in_channels, 256, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(256, 512, kernel_size=3, stride=2, padding=1)]

        # Block 7
        # 10,10,512 -> 10,10,128 -> 5,5,256
        layers += [nn.Conv2d(512, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)]

        # Block 8
        # 5,5,256 -> 5,5,128 -> 3,3,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]

        # Block 9
        # 3,3,256 -> 3,3,128 -> 1,1,256
        layers += [nn.Conv2d(256, 128, kernel_size=1, stride=1)]
        layers += [nn.Conv2d(128, 256, kernel_size=3, stride=1)]
    else:
        layers += [InvertedResidual(in_channels, 512, stride=2, expand_ratio=0.2)]
        layers += [InvertedResidual(512, 256, stride=2, expand_ratio=0.25)]
        layers += [InvertedResidual(256, 256, stride=2, expand_ratio=0.5)]
        layers += [InvertedResidual(256, 64, stride=2, expand_ratio=0.25)]

    return nn.ModuleList(layers)


# 下采样模块
def Downsampling():
    layers = []
    in_channels = 3

    # 300->38
    # pool1 = nn.MaxPool2d(kernel_size=8, stride=8, ceil_mode=True)
    pool1 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    pool2 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    pool3 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

    # 38->19
    pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
    # 19->10
    pool5 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)
    # 10->5
    pool6 = nn.MaxPool2d(kernel_size=2, stride=2)

    layers += [pool1, pool2, pool3, pool4, pool5, pool6]

    return nn.Sequential(*layers)

class ConvBlock(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(ConvBlock, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x

class LFIP(nn.Module):

    def __init__(self, in_planes):
        super(LFIP, self).__init__()
        self.iter_ds = Iter_Downsample()
        self.lcb1 = nn.Sequential(
            ConvBlock(in_planes, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 128, kernel_size=1, stride=1),
            ConvBlock(128, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 512, kernel_size=1, relu=False))
        self.lcb2 = nn.Sequential(
            ConvBlock(in_planes, 256, kernel_size=(3, 3), padding=1), ConvBlock(256, 256, kernel_size=1),
            ConvBlock(256, 256, kernel_size=(3, 3), padding=1), ConvBlock(256, 256, kernel_size=1, stride=1),
            ConvBlock(256, 256, kernel_size=(3, 3), padding=1), ConvBlock(256, 1024, kernel_size=1, relu=False))
        self.lcb3 = nn.Sequential(
            ConvBlock(in_planes, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 128, kernel_size=1),
            ConvBlock(128, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 128, kernel_size=1),
            ConvBlock(128, 128, kernel_size=(3, 3), padding=1), ConvBlock(128, 512, kernel_size=1, relu=False))
        self.lcb4 = nn.Sequential(
            ConvBlock(in_planes, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(3, 3),  padding=1), ConvBlock(64, 64, kernel_size=1),
            ConvBlock(64, 64, kernel_size=(3, 3), padding=1), ConvBlock(64, 256, kernel_size=1, relu=False))

    def forward(self, x):
        img1, img2, img3, img4 = self.iter_ds(x)
        s1 = self.lcb1(img1)
        s2 = self.lcb2(img2)
        s3 = self.lcb3(img3)
        s4 = self.lcb4(img4)
        return s1, s2, s3, s4

class Iter_Downsample(nn.Module):

    def __init__(self,):
        super(Iter_Downsample, self).__init__()
        self.init_ds = nn.Sequential(
            nn.MaxPool2d(kernel_size=2, stride=2, padding=0), nn.MaxPool2d(kernel_size=2, stride=2, padding=0))
        self.ds1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.ds2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.ds3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=1)
        self.ds4 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        x = self.init_ds(x)
        x1 = self.ds1(x)
        x2 = self.ds2(x1)
        x3 = self.ds3(x2)
        x4 = self.ds4(x3)
        return x1, x2, x3, x4

class FAM(nn.Module):

    def __init__(self, plane1, plane2, bn=True, ds=True, att=True):
        super(FAM, self).__init__()
        self.att = att
        self.bn = nn.BatchNorm2d(plane2, eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.dsc = ConvBlock(plane1, plane2, kernel_size=(3, 3), stride=2, padding=1, relu=False) if ds else None
        self.down = ConvBlock(plane2, plane2//2, kernel_size=(1, 1), stride=1)
        self.merge = ConvBlock(plane2, plane2, kernel_size=(3, 3), stride=1, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, o, s, p):

        # o_bn = self.bn(o) if self.bn is not None else o
        # s_bn = s
        # if self.att:
        #     x = o_bn * s_bn + self.dsc(p) if self.dsc is not None else o_bn * s_bn
        # else:
        #     x = o_bn + self.dsc(p) if self.dsc is not None else o_bn
        # out = self.merge(self.relu(x))
        #
        # return out

        o_bn = self.bn(o) if self.bn is not None else o
        s_bn = s
        if self.att:
            # o_bn = self.down(o_bn)
            # s_bn = self.down(s_bn)
            x = o_bn + s_bn + self.dsc(p) if self.dsc is not None else o_bn + s_bn
            # x = torch.cat([o_bn,s_bn], 1) + self.dsc(p) if self.dsc is not None else torch.cat([o_bn,s_bn], 1)

        else:
            x = o_bn + self.dsc(p) if self.dsc is not None else o_bn
        out = self.merge(self.relu(x))

        return out

class SSD300(nn.Module):
    def __init__(self, num_classes, backbone_name, pretrained=False):
        super(SSD300, self).__init__()
        self.num_classes = num_classes
        if backbone_name == "vgg":
            self.vgg = add_vgg(pretrained)
            self.extras = add_extras(1024, backbone_name)
            self.L2Norm = L2Norm(512, 20)

            self.lfip = LFIP(in_planes=3)
            self.Downsampling = Downsampling()

            self.fam1 = FAM(plane1=512, plane2=512, bn=True, ds=False, att=True)
            self.fam2 = FAM(plane1=512, plane2=1024, bn=True, ds=True, att=True)
            self.fam3 = FAM(plane1=1024, plane2=512, bn=False, ds=True, att=True)
            self.fam4 = FAM(plane1=512, plane2=256, bn=False, ds=True, att=False)

            self.eca_attention1 = eca_block(512)
            self.eca_attention2 = eca_block(1025)
            self.eca_attention3 = eca_block(512)
            self.eca_attention4 = eca_block(256)

            self.primary_conv1 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

            self.primary_conv2 = nn.Sequential(
                nn.Conv2d(1024, 512, kernel_size=1, stride=1),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )

            self.primary_conv3 = nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

            self.primary_conv4 = nn.Sequential(
                nn.Conv2d(256, 128, kernel_size=1, stride=1),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

            self.cheap_operation1 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

            self.cheap_operation2 = nn.Sequential(
                nn.Conv2d(512, 512, kernel_size=3, padding=1, groups=512, bias=False),
                nn.BatchNorm2d(512),
                nn.ReLU(inplace=True)
            )

            self.cheap_operation3 = nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1, groups=256, bias=False),
                nn.BatchNorm2d(256),
                nn.ReLU(inplace=True)
            )

            self.cheap_operation4 = nn.Sequential(
                nn.Conv2d(128, 128, kernel_size=3, padding=1, groups=128, bias=False),
                nn.BatchNorm2d(128),
                nn.ReLU(inplace=True)
            )

            self.down_convg1 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(512, 1024, kernel_size=1, stride=1),
                nn.PReLU(1024)
            )

            self.NETconv2d1 = nn.ConvTranspose2d(1024, 512, 2, 2)


            self.down_convg3 = nn.Sequential(
                nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True),
                nn.Conv2d(512, 256, kernel_size=1, stride=1),
                nn.PReLU(256)
            )

            self.NETconv2d3 = nn.ConvTranspose2d(256, 512, 2, 2)

            self.pool1234 = nn.MaxPool2d(kernel_size=2, stride=2, ceil_mode=True)

            mbox = [4, 6, 6, 6, 4, 4]

            loc_layers = []
            conf_layers = []
            backbone_source = [21, -2]
            # ---------------------------------------------------#
            #   在add_vgg获得的特征层里
            #   第21层和-2层可以用来进行回归预测和分类预测。
            #   分别是conv4-3(38,38,512)和conv7(19,19,1024)的输出
            # ---------------------------------------------------#
            for k, v in enumerate(backbone_source):
                loc_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.vgg[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            # -------------------------------------------------------------#
            #   在add_extras获得的特征层里
            #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
            #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
            # -------------------------------------------------------------#
            for k, v in enumerate(self.extras[1::2], 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
        else:
            self.mobilenet = mobilenet_v2(pretrained).features
            self.extras = add_extras(1280, backbone_name)
            self.L2Norm = L2Norm(96, 20)
            mbox = [6, 6, 6, 6, 6, 6]

            loc_layers = []
            conf_layers = []
            backbone_source = [13, -1]
            for k, v in enumerate(backbone_source):
                loc_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(self.mobilenet[v].out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]
            for k, v in enumerate(self.extras, 2):
                loc_layers += [nn.Conv2d(v.out_channels, mbox[k] * 4, kernel_size=3, padding=1)]
                conf_layers += [nn.Conv2d(v.out_channels, mbox[k] * num_classes, kernel_size=3, padding=1)]

        self.loc = nn.ModuleList(loc_layers)
        self.conf = nn.ModuleList(conf_layers)
        self.backbone_name = backbone_name

    def forward(self, x):
        # ---------------------------#
        #   x是300,300,3
        # ---------------------------#
        sources = list()
        loc = list()
        conf = list()

        # generate image pyramid
        s1, s2, s3, s4 = self.lfip(x)

        a1 = self.primary_conv1(s1)
        b1 = self.cheap_operation1(a1)
        c1 = torch.cat([a1,b1], 1)
        s1 = c1 + s1

        a2 = self.primary_conv2(s2)
        b2 = self.cheap_operation2(a2)
        c2 = torch.cat([a2,b2], 1)
        s2 = c2 + s2

        a3 = self.primary_conv3(s3)
        b3 = self.cheap_operation3(a3)
        c3 = torch.cat([a3,b3], 1)
        s3 = c3 + s3

        a4 = self.primary_conv4(s4)
        b4 = self.cheap_operation4(a4)
        c4 = torch.cat([a4,b4], 1)
        s4 = c4 + s4

        # ---------------------------#
        #   获得conv4_3的内容
        #   shape为38,38,512
        # ---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23):
                x = self.vgg[k](x)
        else:
            for k in range(14):
                x = self.mobilenet[k](x)
        # ---------------------------#
        #   conv4_3的内容
        #   需要进行L2标准化
        # ---------------------------#
        x = self.L2Norm(x)
        f1 = self.fam1(x, s1, None)
        sources.append(f1)

        # ---------------------------#
        #   获得conv7的内容
        #   shape为19,19,1024
        # ---------------------------#
        if self.backbone_name == "vgg":
            for k in range(23, len(self.vgg)):
                x = self.vgg[k](x)
        else:
            for k in range(14, len(self.mobilenet)):
                x = self.mobilenet[k](x)
        f2 = self.fam2(x, s2, f1)
        sources.append(f2)

        # -------------------------------------------------------------#
        #   在add_extras获得的特征层里
        #   第1层、第3层、第5层、第7层可以用来进行回归预测和分类预测。
        #   shape分别为(10,10,512), (5,5,256), (3,3,256), (1,1,256)
        # -------------------------------------------------------------#
        for k, v in enumerate(self.extras):
            x = F.relu(v(x), inplace=True)
            if self.backbone_name == "vgg":
                if k == 1:
                    f3 = self.fam3(x, s3, f2)
                    sources.append(f3)
                elif k == 3:
                    f4 = self.fam4(x, s4, f3)
                    sources.append(f4)
                elif k % 2 == 1:
                    sources.append(x)
            else:
                sources.append(x)


        # 4个有效特征层
        s1 = sources[0]
        s2 = sources[1]
        s3 = sources[2]
        s4 = sources[3]

        s1 = self.eca_attention1(s1)
        s2 = self.eca_attention2(s2)
        s3 = self.eca_attention3(s3)
        s4 = self.eca_attention4(s4)


        g1 = self.down_convg1(s1) # 上采样10->38
        outlayer1 = torch.nn.Sigmoid()  # sigmoid
        g1 = outlayer1(g1)  # sigmoid

        g3 = self.down_convg3(s3)  # 上采样10->38
        outlayer2 = torch.nn.Sigmoid()  # sigmoid
        g3 = outlayer2(g3)  # sigmoid

        pes12 = s2 * g1
        p2 = s2 + pes12
        pes12 = self.NETconv2d1(pes12)
        p1 = s1 - pes12

        pes34 = s4 * g3
        p4 = s4 + pes34
        pes34 = self.NETconv2d3(pes34)
        p3 = s3 - pes34

        #
        p1 = self.eca_attention1(p1)
        p2 = self.eca_attention2(p2)
        p3 = self.eca_attention3(p3)
        p4 = self.eca_attention4(p4)

        sources[0] = p1
        sources[1] = p2
        sources[2] = p3
        sources[3] = p4

        # -------------------------------------------------------------#
        #   为获得的6个有效特征层添加回归预测和分类预测
        # -------------------------------------------------------------#
        for (x, l, c) in zip(sources, self.loc, self.conf):
            loc.append(l(x).permute(0, 2, 3, 1).contiguous())
            conf.append(c(x).permute(0, 2, 3, 1).contiguous())

        # -------------------------------------------------------------#
        #   进行reshape方便堆叠
        # -------------------------------------------------------------#
        loc = torch.cat([o.view(o.size(0), -1) for o in loc], 1)
        conf = torch.cat([o.view(o.size(0), -1) for o in conf], 1)
        # -------------------------------------------------------------#
        #   loc会reshape到batch_size, num_anchors, 4
        #   conf会reshap到batch_size, num_anchors, self.num_classes
        # -------------------------------------------------------------#
        output = (
            loc.view(loc.size(0), -1, 4),
            conf.view(conf.size(0), -1, self.num_classes),
        )
        return output

# if __name__ == '__main__':
#     net =
