import paddle
import paddle.nn as nn
import paddle.nn.functional as F

from ppdet.core.workspace import register, serializable
from ppdet.modeling.ops import batch_norm, mish
from ..shape_spec import ShapeSpec

__all__ = ['CSPDarknet']

import paddle
from paddle import nn


class SiLU(nn.Layer):
    @staticmethod
    def forward(x):
        return x * nn.functional.sigmoid(x)


def get_activation(name="silu", inplace=True):
    # paddle does not have inplace !!!
    if name == "silu":
        module = nn.Silu()
    elif name == "relu":
        module = nn.ReLU()
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module

#CBA封装 conv+bn+act
class BaseConv(nn.Layer):
    """ [Conv2d]-[BN]-[activation] """

    def __init__(self, in_channels, out_channels,
                 kernel_size,
                 stride,
                 groups=1,
                 bias=False,
                 activation="silu"):
        super().__init__()
        pad = (kernel_size - 1) // 2
        self.conv = nn.Conv2D(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=pad,
            groups=groups,
            bias_attr=bias,
        )

        weight_attr = paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0), trainable=True)
        bias_attr = paddle.ParamAttr(regularizer=paddle.regularizer.L2Decay(0), trainable=True)
        self.bn = nn.BatchNorm2D(num_features=out_channels, epsilon=1e-3, momentum=0.97, weight_attr=weight_attr, bias_attr=bias_attr)
        self.act = get_activation(activation, inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.act(x)
        return x

    def fuse_forward(self, x):
        x = self.conv(x)
        x = self.act(x)
        return x


# 深度可分离卷积
class DWConv(nn.Layer):
    """ Depthwise Conv + Conv """

    def __init__(self, in_channels, out_channels,
                 kernel_size, stride=1, activation="silu"):
        super().__init__()

        self.dconv = BaseConv(in_channels=in_channels,
                              out_channels=in_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              groups=in_channels,
                              activation=activation)

        self.pconv = BaseConv(in_channels=in_channels,
                              out_channels=out_channels,
                              kernel_size=1,
                              stride=1,
                              groups=1,
                              activation=activation)

    def forward(self, x):
        x = self.dconv(x)
        x = self.pconv(x)
        return x


#瓶颈模块
class Bottleneck(nn.Layer):
    def __init__(self, in_channels, out_channels,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 activation='silu'):

        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv_1 = BaseConv(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1,
                               stride=1,
                               activation=activation)

        Conv = DWConv if depthwise else BaseConv
        self.conv_2 = Conv(in_channels=hidden_channels,
                           out_channels=out_channels,
                           kernel_size=3,
                           stride=1,
                           activation=activation)
        self.use_add = shortcut and in_channels == out_channels

        # print(depthwise, expansion, shortcut, activation)
        self.depthwise = depthwise

    def forward(self, x, flag=""):
        x = self.conv_1(x)

        y = self.conv_2(x)

        if self.use_add:
            y = y + x
        return y


# Resnet层
class ResLayer(nn.Layer):
    def __init__(self, in_channels):
        super().__init__()
        mid_channels = in_channels // 2

        self.layer_1 = BaseConv(in_channels=in_channels,
                                out_channels=mid_channels,
                                kernel_size=1,
                                stride=1,
                                activation='lrelu')

        self.layer_2 = BaseConv(in_channels=mid_channels,
                                out_channels=in_channels,
                                kernel_size=3,
                                stride=1,
                                activation='lrelu')

    def forward(self, x):
        y = self.layer_2(self.layer_1(x))
        return x + y


# YoloV3的SPP层
class SPPBottleneck(nn.Layer):
    """ Spatial Pyramid Pooling - in YOLOv3-SPP """
    def __init__(self, in_channels, out_channels,
                 kernel_sizes=(5, 9, 13),
                 activation='silu'):
        super().__init__()
        hidden_channels = in_channels // 2
        self.conv_1 = BaseConv(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1,
                               stride=1,
                               activation=activation)
        m = [nn.MaxPool2D(kernel_size=ks,
                          stride=1,
                          padding=ks//2) for ks in kernel_sizes] #最大池化+paddin+stride保证输出维度一致
        self.m = nn.LayerList(m)
        conv2_channels = hidden_channels * (len(kernel_sizes) + 1)
        self.conv_2 = BaseConv(in_channels=conv2_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               activation=activation)

    def forward(self, x):
        x = self.conv_1(x)
        x = paddle.concat([x] + [m(x) for m in self.m], axis=1)
        x = self.conv_2(x)
        return x

#CSP层
class CSPLayer(nn.Layer):
    """ C3 in YOLOv5, CSP Bottleneck with 3 conv """
    def __init__(self, in_channels, out_channels,
                 bottleneck_cnt=1,
                 shortcut=True,
                 expansion=0.5,
                 depthwise=False,
                 activation='silu'):
        super().__init__()
        hidden_channels = int(out_channels * expansion)

        self.conv_1 = BaseConv(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1,
                               stride=1,
                               activation=activation)

        self.conv_2 = BaseConv(in_channels=in_channels,
                               out_channels=hidden_channels,
                               kernel_size=1,
                               stride=1,
                               activation=activation)

        self.conv_3 = BaseConv(in_channels=2*hidden_channels,
                               out_channels=out_channels,
                               kernel_size=1,
                               stride=1,
                               activation=activation)

        m = [Bottleneck(in_channels=hidden_channels,
                        out_channels=hidden_channels,
                        shortcut=shortcut,
                        expansion=1.0,
                        depthwise=depthwise,
                        activation=activation) for _ in range(bottleneck_cnt)]

        self.m = nn.Sequential(*m)


    def forward(self, x, flag=""):
        x1 = self.conv_1(x)
        x2 = self.conv_2(x)

        for i in range(len(self.m)):
            if flag == "debug":
                x1 = self.m[i](x1, flag if i == 3 else "")
            else:
                x1 = self.m[i](x1)

        x = paddle.concat((x1, x2), axis=1)

        out = self.conv_3(x)

        return out

#Focus 模块主要是实现没有信息丢失的下采样
class Focus(nn.Layer):
    """ Focus width and height information into channel space """
    def __init__(self, in_channels, out_channels,
                 kernel_size=1, stride=1, activation='silu'):
        super().__init__()
        self.conv = BaseConv(in_channels=in_channels*4,
                             out_channels=out_channels,
                             kernel_size=kernel_size,
                             stride=stride,
                             activation=activation)

    def forward(self, x):
        patch_top_left = x[:, :, ::2, ::2]
        patch_top_right = x[:, :, ::2, 1::2]
        patch_bot_left = x[:, :, 1::2, ::2]
        patch_bot_right = x[:, :, 1::2, 1::2]
        x = paddle.concat((patch_top_left,
                           patch_bot_left,
                           patch_top_right,
                           patch_bot_right), axis=1)
        return self.conv(x)


#yolox暂只用到了CSPDarknet
class CSPDarknet(nn.Layer):
    def __init__(self, dep_mul, wid_mul,
                 out_features=("dark3", "dark4", "dark5"),
                 depthwise=False,
                 activation="silu"):

        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64 * 1.25
        base_depth = max(round(dep_mul * 3), 1)  # 3

        self.stem = Focus(in_channels=3,
                          out_channels=base_channels,
                          kernel_size=3,
                          activation=activation)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(in_channels=base_channels,
                 out_channels=base_channels*2,
                 kernel_size=3,
                 stride=2,
                 activation=activation),
            CSPLayer(in_channels=base_channels*2,
                     out_channels=base_channels*2,
                     bottleneck_cnt=base_depth,
                     depthwise=depthwise,
                     activation=activation)
        )


        # dark3
        self.dark3 = nn.Sequential(
            Conv(in_channels=base_channels*2,
                 out_channels=base_channels*4,
                 kernel_size=3,
                 stride=2,
                 activation=activation),
            CSPLayer(in_channels=base_channels*4,
                     out_channels=base_channels*4,
                     bottleneck_cnt=base_depth*3,
                     depthwise=depthwise,
                     activation=activation)
        )

        #dark 4
        self.dark4 = nn.Sequential(
            Conv(in_channels=base_channels*4,
                 out_channels=base_channels*8,
                 kernel_size=3,
                 stride=2,
                 activation=activation),
            CSPLayer(in_channels=base_channels*8,
                     out_channels=base_channels*8,
                     bottleneck_cnt=base_depth*3,
                     depthwise=depthwise,
                     activation=activation)
        )


        #dark 5
        self.dark5 = nn.Sequential(
            Conv(in_channels=base_channels*8,
                 out_channels=base_channels*16,
                 kernel_size=3,
                 stride=2,
                 activation=activation),
            SPPBottleneck(in_channels=base_channels*16,
                          out_channels=base_channels*16,
                          activation=activation),
            CSPLayer(in_channels=base_channels*16,
                     out_channels=base_channels*16,
                     bottleneck_cnt=base_depth,
                     shortcut=False,
                     depthwise=depthwise,
                     activation=activation)
        )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)

        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)

        outputs["dark3"] = x
        x = self.dark4(x)

        outputs["dark4"] = x
        x = self.dark5(x)

        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
