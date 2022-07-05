"""
@Time ：2022/7/4 21:54
@Auth ：那时那月那人
@MAIL：1312759081@qq.com
"""

import numpy as np


def convert_to_xywh(boxes):
    """
    x1,y1,x2,y2 -> x_center,ycenter,w,h
    :param boxes:
    :return:
    """

    return np.concatenate(
        [(boxes[..., :2] + boxes[..., 2:]) / 2.0, boxes[..., 2:] - boxes[..., :2]],
        axis=-1,
    )


def convert_to_corners(boxes):
    """
    xywh -> x1y1x2y2
    :param boxes:
    :return:
    """
    return np.concatenate(
        [boxes[..., :2] - boxes[..., 2:] / 2.0, boxes[..., :2] + boxes[..., 2:] / 2.0],
        axis=-1,

    )


def box_overlap(boxes, query_boxes):
    lu = np.maximum(boxes[:, None, :2], query_boxes[:, :2])
    rd = np.minimum(boxes[:, None, 2:], query_boxes[:, 2:])
    intersection = np.maximum(0.0, rd - lu)
    intersection_area = intersection[:, :, 0] * intersection[:, :, 1]
    boxes_area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    query_boxes_area = (query_boxes[:, 2] - query_boxes[:, 0]) * (query_boxes[:, 3] - query_boxes[:, 1])
    union_area = np.maximum(
        boxes_area[:, None] + query_boxes_area - intersection_area, 1e-8
    )
    return np.clip(intersection_area / union_area, 0.0, 1.0)


def box_transform(ex_rois, gt_rois):
    """
    box regression formula
    :param ex_rois:
    :param gt_rois:
    :return:
    """
    ex_xywh = convert_to_xywh(ex_rois)
    gt_xywh = convert_to_xywh(gt_rois)
    dxy = (gt_xywh[:, 0:2] - ex_xywh[:, 0:2]) / gt_xywh[:, 2:4]
    dwh = np.log(gt_xywh[:, 2:4] / (ex_xywh[:, 2:4] + 1e-6))
    dxywh = np.concatenate([dxy, dwh], axis=-1)
    return dxywh


def bbox_transform_inv(boxes, deltas):
    boxes = boxes.astype(deltas.dtype, copy=False)

    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    dx = deltas[:, 0]
    dy = deltas[:, 1]
    dw = deltas[:, 2]
    dh = deltas[:, 3]

    pred_ctr_x = dx * widths + ctr_x
    pred_ctr_y = dy * heights + ctr_y
    pred_w = np.exp(dw) * widths
    pred_h = np.exp(dh) * heights

    pred_boxes = np.zeros(deltas.shape, dtype=deltas.dtype)
    # x1
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * pred_w
    # y1
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * pred_h
    # x2
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * pred_w
    # y2
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * pred_h

    return pred_boxes
