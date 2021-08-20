#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) 2014-2021 Megvii Inc. All rights reserved.

import numpy as np

import paddle


__all__ = [
    "filter_box",
    "postprocess",
    "bboxes_iou",
    "matrix_iou",
    "adjust_box_anns",
    "xyxy2xywh",
    "xyxy2cxcywh",
]


def filter_box(output, scale_range):
    """
    output: (N, 5+class) shape
    """
    min_scale, max_scale = scale_range
    w = output[:, 2] - output[:, 0]
    h = output[:, 3] - output[:, 1]
    keep = (w * h > min_scale * min_scale) & (w * h < max_scale * max_scale)
    return output[keep]

def nms(boxes, scores, nms_thr):
    """Single class NMS implemented in Numpy."""
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= nms_thr)[0]
        order = order[inds + 1]

    return keep


def multiclass_nms(boxes, scores, nms_thr, score_thr):
    """Multiclass NMS implemented in Numpy"""
    final_dets = []

    num_classes = 81
    for cls_ind in range(1, num_classes):
        cls_scores = scores[scores[:, 1] == cls_ind]
        cls_box = boxes[scores[:, 1] == cls_ind]
        valid_score_mask = cls_scores[:, 0] > score_thr
        if valid_score_mask.sum() == 0:
            continue
        else:
            valid_scores = cls_scores[valid_score_mask]
            valid_boxes = cls_box[valid_score_mask]
            keep = nms(valid_boxes, valid_scores[:, 0], nms_thr)
            if len(keep) > 0:
                dets = np.concatenate([valid_boxes[keep], valid_scores[keep, :]], 1)
                final_dets.append(dets)
    if len(final_dets) == 0:
        return None
    return np.concatenate(final_dets, 0)

@paddle.no_grad()
def postprocess_bk(prediction, num_classes, conf_thre=0.45, nms_thre=0.45):
    box_corner = paddle.zeros_like(prediction)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2



    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):
        if not image_pred.shape[0]:
            continue

        class_conf = paddle.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        class_pred = paddle.argmax(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()

        detections = paddle.concat(
            (image_pred[:, :5].astype('float'), class_conf.astype('float'), class_pred.astype('float')), 1)
        conf_mask_ind = conf_mask.nonzero()
        detections = paddle.gather(detections, conf_mask_ind)
        if not detections.shape[0]:
            continue
        detections = detections.numpy()

        score = [(detections[:, 4] * detections[:, 5])[:, None], detections[:, 6][:, None]]
        score = np.concatenate(score, 1)
        # print(detections[:, :4], score)
        # print("==============================>")
        output = multiclass_nms(
            detections[:, :4],
            score,
            nms_thre,
            conf_thre
        )

    # print(output.shape, output)

    del prediction
    return output


@paddle.no_grad()
def postprocess(prediction, im_shape, scale_factor, num_classes=80,  conf_thre=0.25, nms_thre=0.45):
    # box_corner = paddle.zeros_like(prediction)
    # box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    # box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    # box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    # box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    # prediction[:, :, :4] = box_corner[:, :, :4]

    conf_thre = 0.001

    pred_npy = prediction.numpy()
    tmp_npy = np.zeros(pred_npy[:, :, :4].shape)
    tmp_npy[:, :, 0] = pred_npy[:, :, 0] - pred_npy[:, :, 2] / 2
    tmp_npy[:, :, 1] = pred_npy[:, :, 1] - pred_npy[:, :, 3] / 2
    tmp_npy[:, :, 2] = pred_npy[:, :, 0] + pred_npy[:, :, 2] / 2
    tmp_npy[:, :, 3] = pred_npy[:, :, 1] + pred_npy[:, :, 3] / 2
    # prediction[:, :, :4] = paddle.to_tensor(tmp_npy).astype(prediction.dtype)
    pred_npy[:, :, :4] = tmp_npy

    scale_factor = scale_factor.numpy()

    boxes = []
    boxes_num = []


    for i in range(pred_npy.shape[0]):
        if not pred_npy[i].shape[0]:
            continue

        # class_conf = paddle.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)
        # class_pred = paddle.argmax(image_pred[:, 5: 5 + num_classes], 1, keepdim=True) #checked
        # conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        #
        # detections = paddle.concat(
        #     (image_pred[:, :5].astype('float'), class_conf.astype('float'), class_pred.astype('float')), 1)
        # conf_mask_ind = conf_mask.nonzero()
        # detections = paddle.gather(detections, conf_mask_ind)

        class_conf = np.max(pred_npy[i][:, 5: 5 + num_classes], 1, keepdims = True)
        class_pred = np.argmax(pred_npy[i][:, 5: 5 + num_classes], 1).reshape(class_conf.shape)
        conf_mask = (pred_npy[i][:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        conf_mask_ind = conf_mask.nonzero()
        detections = np.concatenate((pred_npy[i][:, :5].astype("float32"), class_conf.astype("float32"),
                                    class_pred.astype("float32")), axis = 1)[conf_mask_ind]

        if not detections.shape[0]:
            continue
        # detections = detections.numpy()

        score = [(detections[:, 4] * detections[:, 5])[:, None], detections[:, 6][:, None]]
        score = np.concatenate(score, 1)

        output = multiclass_nms(
            detections[:, :4],
            score,
            nms_thre,
            conf_thre
        ) # 输出box + score + cls

        if output is None:
            continue

        #resize
        ratio = np.array([[scale_factor[i][0], scale_factor[i][1], scale_factor[i][0], scale_factor[i][1]]])
        resutls = np.zeros_like(output)
        resutls[:, 0] = output[:, 5]
        resutls[:, 1] = output[:, 4]
        resutls[:, 2:6] = output[:, :4] / ratio

        # print(output)
        # print(resutls)

        #todo 检查box边界
        # if boxes is None:
        #     boxes = resutls
        # else:
        #     boxes = np.concatenate((boxes, resutls))
        boxes += resutls.tolist()
        boxes_num.append(resutls.shape[0])

    del prediction
    return np.asarray(boxes), boxes_num


def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = paddle.maximum(bboxes_a[:, :2].unsqueeze(1), bboxes_b[:, :2])
        br = paddle.minimum(bboxes_a[:, 2:].unsqueeze(1), bboxes_b[:, 2:])
        area_a = paddle.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = paddle.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl = paddle.maximum(
            (bboxes_a[:, :2].unsqueeze(1) - bboxes_a[:, 2:].unsqueeze(1) / 2),
            (bboxes_b[:, :2] - bboxes_b[:, 2:] / 2),
        )
        br = paddle.minimum(
            (bboxes_a[:, :2].unsqueeze(1) + bboxes_a[:, 2:].unsqueeze(1) / 2),
            (bboxes_b[:, :2] + bboxes_b[:, 2:] / 2),
        )

        area_a = paddle.prod(bboxes_a[:, 2:], 1)
        area_b = paddle.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).astype(tl.dtype).prod(axis=2)
    area_i = paddle.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a.unsqueeze(1) + area_b - area_i)


def matrix_iou(a, b):
    """
    return iou of a and b, numpy version for data augenmentation
    """
    lt = np.maximum(a[:, np.newaxis, :2], b[:, :2])
    rb = np.minimum(a[:, np.newaxis, 2:], b[:, 2:])

    area_i = np.prod(rb - lt, axis=2) * (lt < rb).all(axis=2)
    area_a = np.prod(a[:, 2:] - a[:, :2], axis=1)
    area_b = np.prod(b[:, 2:] - b[:, :2], axis=1)
    return area_i / (area_a[:, np.newaxis] + area_b - area_i + 1e-12)


def adjust_box_anns(bbox, scale_ratio, padw, padh, w_max, h_max):
    bbox[:, 0::2] = np.clip(bbox[:, 0::2] * scale_ratio + padw, 0, w_max)
    bbox[:, 1::2] = np.clip(bbox[:, 1::2] * scale_ratio + padh, 0, h_max)
    return bbox


def xyxy2xywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    return bboxes


def xyxy2cxcywh(bboxes):
    bboxes[:, 2] = bboxes[:, 2] - bboxes[:, 0]
    bboxes[:, 3] = bboxes[:, 3] - bboxes[:, 1]
    bboxes[:, 0] = bboxes[:, 0] + bboxes[:, 2] * 0.5
    bboxes[:, 1] = bboxes[:, 1] + bboxes[:, 3] * 0.5
    return bboxes
