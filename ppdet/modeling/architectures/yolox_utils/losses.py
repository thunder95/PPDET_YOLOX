import paddle
from paddle import nn


class IOULoss(nn.Layer):
    def __init__(self, reduction='none', loss_type='iou'):
        """
        :param reduction: 'mean'|'sum'
        :param loss_type: 'iou'|'goiu'
        """
        super().__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0] #怎么确保框数量一样

        pred = pred.reshape((-1, 4))
        target = target.reshape((-1, 4))

        tl = paddle.maximum(
            (pred[:, :2] - pred[:, 2:] / 2), #cxcyWH interssects
            (target[:, :2] - target[:, 2:] / 2)
        ) #top left

        br = paddle.minimum(
            (pred[:, :2] + pred[:, 2:] / 2),
            (target[:, :2] + target[:, 2:] / 2)
        ) #bottom right

        area_p = paddle.prod(pred[:, 2:], 1) #面积p
        area_g = paddle.prod(target[:, 2:], 1) #面积q

        en = paddle.cast((tl < br), tl.dtype).prod(axis=1) #判断是否相交
        area_i = paddle.prod(br - tl, 1) * en #dw * dy
        iou = (area_i) / (area_p + area_g - area_i + 1e-16)

        if self.loss_type == 'iou':
            loss = 1. - iou ** 2 #why 平方
        elif self.loss_type == 'giou':
            c_tl = paddle.minimum(
                (pred[:, :2] - pred[:, 2:] / 2),
                (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = paddle.maximum(
                (pred[:, :2] + pred[:, 2:] / 2),
                (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = paddle.prod(c_br - c_tl, 1)

            # torch.clamp -> paddle.clip [?]
            giou = iou - (area_c - area_i) / area_c.clip(1e-16)
            loss = 1. - giou.clip(min=-1.0, max=1.0)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        return loss