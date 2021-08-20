from paddle import nn
from .network_blocks import BaseConv, CSPLayer, \
    DWConv, Focus, ResLayer, SPPBottleneck



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
        x = self.dark2[0](x)

        x = self.dark2[1](x)
        outputs["dark2"] = x
        x = self.dark3(x)

        outputs["dark3"] = x
        x = self.dark4(x)

        outputs["dark4"] = x
        x = self.dark5(x)

        outputs["dark5"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features} # 需要检查下， items
