from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from ppdet.core.workspace import register, create
from .meta_arch import BaseArch

from .yolox_utils.yolo_head import YoloXHead
from .yolox_utils.yolo_pafpn import YoloPAFPN
from .yolox_utils.boxes import postprocess
import paddle
import numpy as np

__all__ = ['YOLOX']

@register
class YOLOX(BaseArch):
    __category__ = 'architecture'
    __shared__ = ['data_format']
    # __inject__ = ['post_process']

    def __init__(self, data_format='NCHW'):
        super(YOLOX, self).__init__(data_format=data_format)
        self.backbone = YoloPAFPN()
        self.head = YoloXHead(num_classes=80)

    @classmethod
    def from_config(cls, cfg, *args, **kwargs):
        return {}


    def _forward(self):
        # input_img = self.inputs["image"] #输入格式不一样

        input_img = paddle.to_tensor(np.load("/f/tmp_rida_report/input_img.npy"))
        targets = paddle.to_tensor(np.load("/f/tmp_rida_report/input_label.npy"))

        fpn_outs = self.backbone(input_img)
        # print("===>", self.inputs["gt_class"].shape, self.inputs["gt_bbox"].shape)
        # targets = paddle.concat((self.inputs["gt_class"], self.inputs["gt_bbox"]), axis=2)
        # print("====>", self.inputs["im_id"])

        if self.training:
            assert targets is not None
            loss, iou_loss, conf_loss, cls_loss, l1_loss, num_fg = self.head(
                fpn_outs, targets, input_img
            )

            return {
                "total_loss": loss,
                "iou_loss": iou_loss,
                "l1_loss": l1_loss,
                "conf_loss": conf_loss,
                "cls_loss": cls_loss,
                "num_fg": num_fg,
            }
            # loss = self.head(
            #     fpn_outs, targets, input_img
            # )
            # return {"total_loss": loss}


            # loss = 0.0
            # mse_loss = paddle.nn.MSELoss()
            # for i in range(len(fpn_outs)):
            #     t = paddle.ones_like(fpn_outs[i]).astype(fpn_outs[i].dtype)
            #     loss += mse_loss(fpn_outs[i], t)
            # print("mse loss: ", loss)
            # return {"total_loss": loss}


        else:
            outputs = self.head(fpn_outs)
            boxes, boxes_num = postprocess(outputs, self.inputs['im_shape'], self.inputs['scale_factor'])
            # print(boxes)
            # print(boxes_num)
            return {'bbox': boxes, 'bbox_num': boxes_num}

    def get_loss(self):
        return self._forward()

    def get_pred(self):
        return self._forward()

