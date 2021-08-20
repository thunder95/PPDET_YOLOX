from paddle import nn
import paddle
from .darknet import CSPDarknet
from .network_blocks import BaseConv, CSPLayer, DWConv


class YoloPAFPN(nn.Layer):
    def __init__(self, depth=1.33, width=1.25,
                 in_features=("dark3", "dark4", "dark5"),
                 in_channels=(256, 512, 1024),
                 depthwise=False,
                 activation="silu", ):

        super().__init__()
        self.backbone = CSPDarknet(depth, width, depthwise=depthwise,
                                   activation=activation)
        self.in_features = in_features
        self.in_channels = in_channels
        Conv = DWConv if depthwise else BaseConv

        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")
        self.lateral_conv0 = BaseConv(in_channels=int(in_channels[2] * width),
                                      out_channels=int(in_channels[1] * width),
                                      kernel_size=1,
                                      stride=1,
                                      activation=activation)

        self.C3_p4 = CSPLayer(in_channels=int(2 * in_channels[1] * width),
                              out_channels=int(in_channels[1] * width),
                              bottleneck_cnt=round(3 * depth),
                              shortcut=False,
                              depthwise=depthwise,
                              activation=activation)  # cat

        self.reduce_conv1 = BaseConv(in_channels=int(in_channels[1] * width),
                                     out_channels=int(in_channels[0] * width),
                                     kernel_size=1,
                                     stride=1,
                                     activation=activation)

        self.C3_p3 = CSPLayer(in_channels=int(2 * in_channels[0] * width),
                              out_channels=int(in_channels[0] * width),
                              bottleneck_cnt=round(3 * depth),
                              shortcut=False,
                              depthwise=depthwise,
                              activation=activation)

        # bottom-up conv
        self.bu_conv2 = Conv(in_channels=int(in_channels[0] * width),
                             out_channels=int(in_channels[0] * width),
                             kernel_size=3,
                             stride=2,
                             activation=activation)

        self.C3_n3 = CSPLayer(in_channels=int(2 * in_channels[0] * width),
                              out_channels=int(in_channels[1] * width),
                              bottleneck_cnt=round(3 * depth),
                              shortcut=False,
                              depthwise=depthwise,
                              activation=activation)

        # bottom-up conv
        self.bu_conv1 = Conv(in_channels=int(in_channels[1] * width),
                             out_channels=int(in_channels[1] * width),
                             kernel_size=3,
                             stride=2,
                             activation=activation)

        self.C3_n4 = CSPLayer(in_channels=int(2 * in_channels[1] * width),
                              out_channels=int(in_channels[2] * width),
                              bottleneck_cnt=round(3 * depth),
                              shortcut=False,
                              depthwise=depthwise,
                              activation=activation)

    def forward(self, input):
        """
        Args:
            inputs: input images.

        Returns:
            Tuple[Tensor]: FPN feature.
            网络结构图参考： https://zhuanlan.zhihu.com/p/397097828
        """

        # input = paddle.to_tensor(paddle.ones((1, 3, 640, 640)))

        #  backbone
        out_features = self.backbone(input)

        features = [out_features[f] for f in self.in_features]
        [x2, x1, x0] = features



        #x0上采样到x1
        fpn_out0 = self.lateral_conv0(x0)  # 1024->512/32
        f_out0 = self.upsample(fpn_out0)  # 512/16
        f_out0 = paddle.concat([f_out0, x1], 1)  # 512->1024/16
        f_out0 = self.C3_p4(f_out0)  # 1024->512/16

        #x1采样到x2
        fpn_out1 = self.reduce_conv1(f_out0)  # 512->256/16
        f_out1 = self.upsample(fpn_out1)  # 256/8
        f_out1 = paddle.concat([f_out1, x2], 1)  # 256->512/8

        #PAN层
        pan_out2 = self.C3_p3(f_out1)  # 512->256/8



        p_out1 = self.bu_conv2(pan_out2)  # 256->256/16
        p_out1 = paddle.concat([p_out1, fpn_out1], 1)  # 256->512/16
        pan_out1 = self.C3_n3(p_out1)  # 512->512/16

        p_out0 = self.bu_conv1(pan_out1)  # 512->512/32
        p_out0 = paddle.concat([p_out0, fpn_out0], 1)  # 512->1024/32
        pan_out0 = self.C3_n4(p_out0)  # 1024->1024/32

        # print("checking...2")
        # import numpy as np
        # print(np.save("/f/tmp_rida_report/paddle_yolox", pan_out0.detach().cpu().numpy()))
        # exit()

        outputs = (pan_out2, pan_out1, pan_out0)
        return outputs
