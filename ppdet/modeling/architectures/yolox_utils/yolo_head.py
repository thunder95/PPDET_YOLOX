import math
from loguru import logger

import paddle
from paddle import nn
import paddle.nn.functional as F

from .boxes import bboxes_iou

from .losses import IOULoss
from .network_blocks import BaseConv, DWConv
import numpy as np

class YoloXHead(nn.Layer):
    def __init__(self, num_classes, width=1.25,
                 strides=(8, 16, 32),
                 in_channels=(256, 512, 1024),
                 activation="silu",
                 depthwise=False):
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = True  # for deploy, set to False

        self.cls_convs = nn.LayerList()
        self.reg_convs = nn.LayerList()
        self.cls_preds = nn.LayerList()
        self.reg_preds = nn.LayerList()
        self.obj_preds = nn.LayerList()
        self.stems = nn.LayerList()
        Conv = DWConv if depthwise else BaseConv

        for i in range(len(in_channels)):
            self.stems.append(BaseConv(in_channels=int(in_channels[i] * width),
                                       out_channels=int(256 * width),
                                       kernel_size=1,
                                       stride=1,
                                       activation=activation, ))

            self.cls_convs.append(nn.Sequential(Conv(in_channels=int(256 * width),
                                                     out_channels=int(256 * width),
                                                     kernel_size=3,
                                                     stride=1,
                                                     activation=activation, ),

                                                Conv(in_channels=int(256 * width),
                                                     out_channels=int(256 * width),
                                                     kernel_size=3,
                                                     stride=1,
                                                     activation=activation, ), ))

            self.reg_convs.append(nn.Sequential(Conv(in_channels=int(256 * width),
                                                     out_channels=int(256 * width),
                                                     kernel_size=3,
                                                     stride=1,
                                                     activation=activation, ),

                                                Conv(in_channels=int(256 * width),
                                                     out_channels=int(256 * width),
                                                     kernel_size=3,
                                                     stride=1,
                                                     activation=activation, ), ))

            self.cls_preds.append(nn.Conv2D(in_channels=int(256 * width),
                                            out_channels=self.n_anchors * self.num_classes,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0))

            self.reg_preds.append(nn.Conv2D(in_channels=int(256 * width),
                                            out_channels=4,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0, ))

            self.obj_preds.append(nn.Conv2D(in_channels=int(256 * width),
                                            out_channels=self.n_anchors * 1,
                                            kernel_size=1,
                                            stride=1,
                                            padding=0, ))

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOULoss(reduction="none")
        self.strides = strides
        self.grids = [paddle.zeros((1,))] * len(in_channels)

    def initialize_biases(self, prior_prob):
        for conv in self.cls_preds:
            b = conv.bias.reshape((self.n_anchors, -1))
            data = b.numpy()
            data.fill(-math.log((1 - prior_prob) / prior_prob))
            conv.bias.set_value(data.flatten())

        for conv in self.obj_preds:
            b = conv.bias.reshape((self.n_anchors, -1))
            data = b.numpy()
            data.fill(-math.log((1 - prior_prob) / prior_prob))
            conv.bias.set_value(data.flatten())

    def forward(self, xin, labels=None, imgs=None):

        # input shape:

        # xin[0]: torch.Size([1, 64, 80, 80])
        # xin[1]: torch.Size([1, 128, 40, 40])
        # xin[2]: torch.Size([1, 256, 20, 20])

        # labels: torch.Size([1, 120, 5])

        # imgs:   torch.Size([1, 3, 640, 640])

        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, (cls_conv, reg_conv, stride_this_level, x) in \
                enumerate(zip(self.cls_convs, self.reg_convs, self.strides, xin)):
            x = self.stems[k](x)
            cls_x = x
            reg_x = x

            cls_feat = cls_conv(cls_x)
            cls_output = self.cls_preds[k](cls_feat)

            reg_feat = reg_conv(reg_x)
            reg_output = self.reg_preds[k](reg_feat)
            obj_output = self.obj_preds[k](reg_feat)

            if self.training:
                output = paddle.concat([reg_output, obj_output, cls_output], 1)
                output, grid = self.get_output_and_grid(
                    output, k, stride_this_level, xin[0].dtype
                )
                x_shifts.append(grid[:, :, 0])
                y_shifts.append(grid[:, :, 1])

                expanded_strides.append((paddle.ones((1, grid.shape[1]))
                                         * stride_this_level).
                                        astype(xin[0].dtype))

                if self.use_l1:
                    batch_size = reg_output.shape[0]
                    hsize, wsize = reg_output.shape[-2:]
                    reg_output = reg_output.reshape((batch_size, self.n_anchors,
                                                     4, hsize, wsize))
                    reg_output = reg_output.transpose(
                        (0, 1, 3, 4, 2)).reshape((batch_size, -1, 4))
                    origin_preds.append(reg_output.clone())

            else:
                output = paddle.concat([reg_output, F.sigmoid(obj_output),
                                        F.sigmoid(cls_output)], 1)

            outputs.append(output)

        if self.training:
            return self.get_losses(
                imgs,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                paddle.concat(outputs, 1),
                origin_preds,
                dtype=xin[0].dtype,
            )
        else:
            self.hw = [x.shape[-2:] for x in outputs]
            # [batch, n_anchors_all, 85]
            outputs = paddle.concat(
                [x.flatten(start_axis=2) for x in outputs], 2
            ).transpose((0, 2, 1))
            if self.decode_in_inference:
                return self.decode_outputs(outputs, dtype=xin[0].dtype)
            else:
                return outputs

    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack((xv, yv), 2).reshape((1,1,hsize,wsize,2)).astype(dtype)
            self.grids[k] = grid

        output = output.reshape((batch_size, self.n_anchors, n_ch, hsize, wsize))
        output = output.transpose((0, 1, 3, 4, 2))\
                       .reshape((batch_size, self.n_anchors * hsize * wsize, -1))
        grid = grid.reshape((1, -1, 2))

        # print(output.shape)

        output[:, :, :2] = (output[:, :, :2] + grid) * stride
        output[:, :, 2:4] = paddle.exp(output[:, :, 2:4]) * stride

        return output, grid

    @paddle.no_grad()
    def decode_outputs(self, outputs, dtype):
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(self.hw, self.strides):
            yv, xv = paddle.meshgrid([paddle.arange(hsize), paddle.arange(wsize)])
            grid = paddle.stack((xv, yv), 2).reshape((1, -1, 2))
            grids.append(grid)
            shape = grid.shape[:2]
            strides.append(paddle.full((*shape, 1), stride))

        grids = paddle.concat(grids, 1).astype(dtype)
        strides = paddle.concat(strides, 1).astype(dtype)

        # grids = grids.expand((outputs.shape[0], grids.shape[1], grids.shape[2]))
        # strides = strides.expand((outputs.shape[0], strides.shape[1], strides.shape[2]))
        #
        # # print(outputs.shape)
        # print(outputs.shape, grids.shape, strides.shape)
        # outputs[:, :, :2] += grids
        # outputs[:, :, :2] = outputs[:, :, :2] * strides
        # outputs[:, :, 2:4] = paddle.exp(outputs[:, :, 2:4]) * strides

        strides_npy = strides.numpy()
        outputs_npy = outputs.numpy()
        grids_npy = grids.numpy()
        outputs_npy[:, :, :2] = (outputs_npy[:, :, :2] + grids_npy) * strides_npy
        outputs_npy[:, :, 2:4] = np.exp(outputs_npy[:, :, 2:4]) * strides_npy
        outputs = paddle.to_tensor(outputs_npy)
        return outputs

    def get_losses(
            self,
            imgs,
            x_shifts,
            y_shifts,
            expanded_strides,
            labels,
            outputs,
            origin_preds,
            dtype,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]                       => [1, 8400, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]           => [1, 8400, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]                    => [1, 8400, 80]

        # calculate targets
        # labels shape: [batch, coords_and_class(85), #objects_in_img]
        mixup = labels.shape[2] > 5
        if mixup:
            label_cut = labels[:, :, :5]
        else:
            label_cut = labels
        nlabel = (label_cut.sum(axis=2) > 0).astype(paddle.int32).sum(axis=1)  # #objects => 21

        total_num_anchors = outputs.shape[1]
        x_shifts = paddle.concat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = paddle.concat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = paddle.concat(expanded_strides, 1)  # [1, n_anchors_all ]
        if self.use_l1:
            origin_preds = paddle.concat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []

        num_fg = 0.0
        num_gts = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])     # 前面的加是在第1维上，这里的第0维是 batch_id
            num_gts += num_gt                   # 计算一共有几个目标
            if num_gt == 0:                     # 没有目标，给 0
                _dtype = outputs.dtype
                cls_target = paddle.zeros((0, self.num_classes), dtype=_dtype)
                reg_target = paddle.zeros((0, 4), dtype=_dtype)
                l1_target = paddle.zeros((0, 4), dtype=_dtype)
                obj_target = paddle.zeros((total_num_anchors, 1), dtype=_dtype)
                fg_mask = paddle.zeros((total_num_anchors, )).astype(paddle.bool)

            else:                               # 有目标，计算
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]   # [21, 4] 从这个 batch 中获取 num_gt 个（也就是标记数量） 1:5是xyxy, 也就是bbox
                gt_classes = labels[batch_idx, :num_gt, 0]              # [21,]   ground truth 分类， gt -> ground truth  0是类别
                bboxes_preds_per_image = bbox_preds[batch_idx]          # 模型输出的预测，这里是在第0维度取 batch_id

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                    )
                except RuntimeError:
                    logger.error(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    # torch.cuda.empty_cache()  # todo: find alternative for paddle
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        imgs,
                        "cpu",
                    )

                # torch.cuda.empty_cache()  # todo: again, emm
                num_fg += num_fg_img

                cls_target = F.one_hot(
                    gt_matched_classes.astype(paddle.int64), self.num_classes
                ) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = paddle.index_select(gt_bboxes_per_image, matched_gt_inds)
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        # outputs.new_zeros((num_fg_img, 4)),
                        paddle.zeros((num_fg_img, 4), dtype=outputs.dtype),
                        reg_target, # gt_bboxes_per_image[matched_gt_inds],
                        paddle.masked_select(expanded_strides[0], fg_mask),
                        x_shifts=paddle.masked_select(x_shifts[0], fg_mask),
                        y_shifts=paddle.masked_select(y_shifts[0], fg_mask),
                    )

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.astype(dtype))
            fg_masks.append(fg_mask)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = paddle.concat(cls_targets, 0) #空标签会报错, 在预处理时干掉, 跟原作者代码不一致
        reg_targets = paddle.concat(reg_targets, 0)
        obj_targets = paddle.concat(obj_targets, 0)
        fg_masks = paddle.concat(fg_masks, 0)
        if self.use_l1:
            l1_targets = paddle.concat(l1_targets, 0)

        num_fg = max(num_fg, 1)

        _mask = fg_masks.tile((4,)).reshape((4, -1)).astype(paddle.int32).t().astype(paddle.bool)

        loss_iou = self.iou_loss( paddle.masked_select(bbox_preds.reshape((-1, 4)), _mask).reshape((-1, 4)), reg_targets)
        loss_iou = loss_iou.sum() / num_fg

        if self.use_l1:
            loss_l1 = (
                        self.l1_loss( paddle.masked_select(origin_preds.reshape((-1, 4)), _mask).reshape((-1, 4)), l1_targets)
                      ).sum() / num_fg
        else:
            loss_l1 = 0.0

        # print(num_fg, np.isnan(obj_preds.numpy()).any(), np.isnan(obj_targets.numpy()).any())
        loss_obj = (
                       self.bcewithlog_loss(obj_preds.reshape((-1, 1)), obj_targets)
                   ).sum() / num_fg

        _mask = fg_masks.tile((self.num_classes,)).reshape((self.num_classes, -1)).astype(paddle.int32).t().astype(paddle.bool)
        loss_cls = (
                       self.bcewithlog_loss(
                           paddle.masked_select(cls_preds.reshape((-1, self.num_classes)), _mask).reshape((-1, self.num_classes)), cls_targets
                       )
                   ).sum() / num_fg


        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        return (
            loss,
            reg_weight * loss_iou,
            loss_obj,
            loss_cls,
            loss_l1,
            num_fg / max(num_gts, 1),
        )

    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = paddle.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = paddle.log(gt[:, 3] / stride + eps)
        return l1_target

    @paddle.no_grad()
    def get_assignments(
            self,
            batch_idx,
            num_gt,
            total_num_anchors,
            gt_bboxes_per_image,
            gt_classes,
            bboxes_preds_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            cls_preds,
            bbox_preds,
            obj_preds,
            labels,
            imgs,
            mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().astype(paddle.float32)
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().astype(paddle.float32)
            gt_classes = gt_classes.cpu().astype(paddle.float32)
            expanded_strides = expanded_strides.cpu().astype(paddle.float32)
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        # fg_mask   => bool tensor of [8400,]
        # bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]  # [8400, 4] => [3600, 4]
        # cls_preds_ = cls_preds[batch_idx][fg_mask]                # [1, 8400, 80] => [3600, 80]
        # obj_preds_ = obj_preds[batch_idx][fg_mask]                # [1, 8400, 1] => [3600, 1]
        # num_in_boxes_anchor = bboxes_preds_per_image.shape[0]     # [3600, ]
        xx, yy = bboxes_preds_per_image.shape
        _mask = fg_mask.astype(paddle.int32).tile((yy, )).reshape((-1, xx)).t().astype(paddle.bool)
        bboxes_preds_per_image = paddle.masked_select(bboxes_preds_per_image, _mask).reshape((-1, yy))
        xx, yy = cls_preds[batch_idx].shape
        _mask = fg_mask.astype(paddle.int32).tile((yy,)).reshape((-1, xx)).t().astype(paddle.bool)
        cls_preds_ = paddle.masked_select(cls_preds[batch_idx], _mask).reshape((-1, yy))
        xx, yy = obj_preds[batch_idx].shape
        _mask = fg_mask.astype(paddle.int32).tile((yy,)).reshape((-1, xx)).t().astype(paddle.bool)
        obj_preds_ = paddle.masked_select(obj_preds[batch_idx], _mask).reshape((-1, yy))
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.astype(paddle.int64), self.num_classes)
                .astype(paddle.float32)
                .unsqueeze(1)
                .tile((1, num_in_boxes_anchor, 1))
        )
        pair_wise_ious_loss = -paddle.log(pair_wise_ious + 1e-8)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        cls_preds_ = (
            F.sigmoid(cls_preds_.astype(paddle.float32).unsqueeze(0).tile((num_gt, 1, 1)))
            * F.sigmoid(obj_preds_.unsqueeze(0).tile((num_gt, 1, 1)))
        )

        pair_wise_cls_loss = F.binary_cross_entropy(
            cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
        ).sum(-1)
        del cls_preds_

        cost = (
                pair_wise_cls_loss
                + 3.0 * pair_wise_ious_loss
                # + 100000.0 * (~is_in_boxes_and_center)
                + 100000.0 * paddle.logical_xor(is_in_boxes_and_center,
                                                paddle.ones_like(is_in_boxes_and_center))
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    def get_in_boxes_info(
            self,
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .tile((num_gt, 1))
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
                .unsqueeze(0)
                .tile((num_gt, 1))
        )

        gt_bboxes_per_image_l = (
            (gt_bboxes_per_image[:, 0] - 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .tile((1, total_num_anchors))
        )
        gt_bboxes_per_image_r = (
            (gt_bboxes_per_image[:, 0] + 0.5 * gt_bboxes_per_image[:, 2])
                .unsqueeze(1)
                .tile((1, total_num_anchors))
        )
        gt_bboxes_per_image_t = (
            (gt_bboxes_per_image[:, 1] - 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .tile((1, total_num_anchors))
        )
        gt_bboxes_per_image_b = (
            (gt_bboxes_per_image[:, 1] + 0.5 * gt_bboxes_per_image[:, 3])
                .unsqueeze(1)
                .tile((1, total_num_anchors))
        )

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = paddle.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(-1) > 0.0
        is_in_boxes_all = is_in_boxes.astype(paddle.int32).sum(0) > 0
        # in fixed center

        center_radius = 2.5

        gt_bboxes_per_image_l = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
            (1, total_num_anchors)
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_r = (gt_bboxes_per_image[:, 0]).unsqueeze(1).tile(
            (1, total_num_anchors)
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_t = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
            (1, total_num_anchors)
        ) - center_radius * expanded_strides_per_image.unsqueeze(0)
        gt_bboxes_per_image_b = (gt_bboxes_per_image[:, 1]).unsqueeze(1).tile(
            (1, total_num_anchors)
        ) + center_radius * expanded_strides_per_image.unsqueeze(0)

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = paddle.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(-1) > 0.0
        is_in_centers_all = is_in_centers.astype(paddle.int32).sum(0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = paddle.logical_or(is_in_boxes_all, is_in_centers_all)

        # A = is_in_boxes[:, is_in_boxes_anchor]
        xx, yy = is_in_boxes.shape
        A = paddle.masked_select(is_in_boxes.astype(paddle.int32), is_in_boxes_anchor.tile((xx, 1))).reshape((xx, -1))

        # B = is_in_centers[:, is_in_boxes_anchor]
        xx, yy = is_in_centers.shape
        B = paddle.masked_select(is_in_centers.astype(paddle.int32), is_in_boxes_anchor.tile((xx, 1))).reshape((xx, -1))

        is_in_boxes_and_center = paddle.logical_and(A.astype(paddle.bool), B.astype(paddle.bool))
        return is_in_boxes_anchor, is_in_boxes_and_center

    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = paddle.zeros_like(cost)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.shape[1])
        topk_ious, _ = paddle.topk(ious_in_boxes_matrix, n_candidate_k, axis=1)
        # dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = paddle.clip(topk_ious.sum(1).astype(paddle.int32), min=1)

        for gt_idx in range(num_gt):
            _, pos_idx = paddle.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx].item(), largest=False
            )

            # matching_matrix[gt_idx][pos_idx] = 1.0
            m = matching_matrix[gt_idx]
            pos_idx_one_hot = F.one_hot(pos_idx, m.shape[-1]).sum(0)
            matching_matrix[gt_idx] = paddle.where(pos_idx_one_hot > 0,
                                                   paddle.ones_like(m), m)

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).astype(paddle.int32).sum() > 0:

            # shapes:
            # cost => [120, 27]
            # (anchor_matching_gt > 1) => [27]

            # matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            # matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            # todo: check gradient!!!

            xx, yy = cost.shape
            _cost = paddle.masked_select(cost, (anchor_matching_gt > 1).tile((xx, 1)))
            cost_argmin = _cost.reshape((xx, -1)).argmin(0)

            xx, yy = matching_matrix.shape
            # assign => select mask + create new + set_value

            # anchor_matching_gt => shape [2833,] , sum => 24
            # matching_matrix => shape [24, 2833]
            xx, yy = matching_matrix.shape
            # matching_matrix[:, anchor_matching_gt > 1] *= 0.0
            matching_matrix = paddle.where((anchor_matching_gt > 1).tile((xx, 1)), paddle.zeros_like(matching_matrix), matching_matrix)
            # matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1.0
            # expect: [7., 8., 8., 6., 6., 3., 3., 4., 5., 0., 6., 2., 4., 7., 5., 3., 2., 6.,
            #         3., 0., 4., 8., 2., 6.])

            axis_y_choose = (anchor_matching_gt > 1)  # shape [2833], sum = 12
            # 12 -> 2833
            _mask_idx = paddle.masked_select(paddle.arange(0, yy), axis_y_choose)
            for i,j in zip(cost_argmin, _mask_idx): matching_matrix[i,j] = 1.0

            # expect: [7., 8., 8., 6., 6., 3., 3., 7., 5., 0., 6.,
            #          7., 7., 8., 5., 3., 2., 6.,
            #         3., 0., 4., 8., 2., 6.]

        # matching_matrix => [6, 2101]
        fg_mask_inboxes = matching_matrix.sum(0) > 0.0              # =>
        num_fg = fg_mask_inboxes.astype(paddle.int32).sum().item()  # =>

        # fg_mask[fg_mask.clone()] = fg_mask_inboxes

        mask_shape, = fg_mask.shape  # 8400
        mask_idx = paddle.masked_select(paddle.masked_select(paddle.arange(0, mask_shape), fg_mask), fg_mask_inboxes)  # [24,]
        fg_mask.set_value(F.one_hot(mask_idx, num_classes=mask_shape).sum(0).astype(paddle.bool))

        # matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        xx, yy = matching_matrix.shape

        matched_gt_inds = (
            paddle.masked_select(
                matching_matrix,
                fg_mask_inboxes.tile((xx, 1))
            ).reshape((xx, -1))
            .argmax(0)
        )

        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = paddle.masked_select((matching_matrix * pair_wise_ious).sum(0),
                                                       fg_mask_inboxes)

        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

