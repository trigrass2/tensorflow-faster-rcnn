import numpy as np


def generate_proposal_boxes(feature_map_size, deltas, scores, im_size, scaled_im_size, config_proposal):
    """filter out invalid boxes"""
    deltas = np.reshape(deltas, [-1, 4])

    anchors = locate_anchors(feature_map_size, config_proposal.feat_stride, config_proposal.anchors)
    pred_boxes = fast_rcnn_bbox_transform_inv(anchors, deltas)
    scale = (np.array([im_size[1], im_size[0], im_size[1], im_size[0]]) - 1.0) / (np.array([scaled_im_size[1], scaled_im_size[0], scaled_im_size[1], scaled_im_size[0]]) - 1.0)
    pred_boxes = (pred_boxes - 1) * scale + 1

    pred_boxes = clip_boxes(pred_boxes, im_size)
    scores = scores[:, 1]

    pred_boxes, scores = filter_boxes(config_proposal.min_box_size, pred_boxes, scores)
    indx = np.argsort(scores)[::-1]
    pred_boxes = pred_boxes[indx]
    scores = scores[indx]

    return pred_boxes, scores


def generate_detection_boxes(deltas, boxes, im_size):
    """
    Args:
        deltas: N x 84
    """
    pred_boxes = []
    for i in range(21):
        b = fast_rcnn_bbox_transform_inv(boxes, deltas[:, i * 4: (i + 1) * 4])
        b = clip_boxes(b, im_size)
        pred_boxes.append(b)

    return np.hstack(pred_boxes)


def clip_boxes(boxes, im_size):
    h, w = im_size
    boxes[:, 0::2] = np.maximum(np.minimum(boxes[:, 0::2], w), 1)
    boxes[:, 1::2] = np.maximum(np.minimum(boxes[:, 1::2], h), 1)
    return boxes


def filter_boxes(min_box_size, boxes, scores):
    w = boxes[:, 2] - boxes[:, 0] + 1
    h = boxes[:, 3] - boxes[:, 1] + 1
    ind = (w >= min_box_size) * (h >= min_box_size)
    boxes = boxes[ind]
    scores = scores[ind]

    return boxes, scores


def locate_anchors(feature_map_size, feat_stride, anchors):
    # feature_map_size, normally tensor.shape = (height, width)
    # however, interpret as (y, x)
    y, x = feature_map_size
    meshgrid = np.mgrid[0:y, 0:x] * feat_stride
    shift_anchors = []
    for i in range(y):
        for j in range(x):
            shift_y, shift_x = meshgrid[:, i, j]
            shift_anchors.append(anchors + np.array([shift_x, shift_y, shift_x, shift_y]))

    return np.asarray(shift_anchors).reshape((-1, 4))


def fast_rcnn_bbox_transform(ex_boxes, gt_boxes):
    eps=1e-10

    ex_widths = ex_boxes[:, 2] - ex_boxes[:, 0] + 1.0
    ex_heights = ex_boxes[:, 3] - ex_boxes[:, 1] + 1.0
    ex_center_x = ex_boxes[:, 0] + 0.5 * (ex_widths - 1)
    ex_center_y = ex_boxes[:, 1] + 0.5 * (ex_heights - 1)

    gt_widths = gt_boxes[:, 2] - gt_boxes[:, 0] + 1.0
    gt_heights = gt_boxes[:, 3] - gt_boxes[:, 1] + 1.0
    gt_center_x = gt_boxes[:, 0] + 0.5 * (gt_widths - 1)
    gt_center_y = gt_boxes[:, 1] + 0.5 * (gt_heights - 1)

    targets_dx = (gt_center_x - ex_center_x) / (ex_widths + eps)
    targets_dy = (gt_center_y - ex_center_y) / (ex_heights + eps)
    targets_dw = np.log(gt_widths / ex_widths)
    targets_dh = np.log(gt_heights / ex_heights)

    regression_label = np.array([targets_dx, targets_dy, targets_dw, targets_dh])
    regression_label = regression_label.reshape((4, -1)).transpose()
    return regression_label


def fast_rcnn_bbox_transform_inv(boxes, box_deltas):
    src_w = boxes[:, 2] - boxes[:, 0] + 1.0
    src_h = boxes[:, 3] - boxes[:, 1] + 1.0
    src_center_x = boxes[:, 0] + 0.5 * (src_w - 1.0)
    src_center_y = boxes[:, 1] + 0.5 * (src_h - 1.0)

    dst_ctr_x = box_deltas[:, 0]
    dst_ctr_y = box_deltas[:, 1]
    dst_scl_x = box_deltas[:, 2]
    dst_scl_y = box_deltas[:, 3]

    pred_ctr_x = dst_ctr_x * src_w + src_center_x
    pred_ctr_y = dst_ctr_y * src_h + src_center_y
    pred_w = np.exp(dst_scl_x) * src_w
    pred_h = np.exp(dst_scl_y) * src_h

    pred_boxes = np.zeros(box_deltas.shape)
    pred_boxes[:, 0] = pred_ctr_x - 0.5 * (pred_w - 1)
    pred_boxes[:, 1] = pred_ctr_y - 0.5 * (pred_h - 1)
    pred_boxes[:, 2] = pred_ctr_x + 0.5 * (pred_w - 1)
    pred_boxes[:, 3] = pred_ctr_y + 0.5 * (pred_h - 1)

    return pred_boxes
