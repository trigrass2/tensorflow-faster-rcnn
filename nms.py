import numpy as np


def nms(aboxes, overlap_thres):
    """
    Args:
        aboxes: N x 5, [boxes, scores]
    Returns:
        pick: M, pick index
    """
    x1 = aboxes[:, 0]
    y1 = aboxes[:, 1]
    x2 = aboxes[:, 2]
    y2 = aboxes[:, 3]
    s = aboxes[:, 4]

    area = (x2 - x1 + 1.0) * (y2 - y1 + 1.0)
    indx = np.argsort(s)

    pick = []
    while np.any(indx):
        i = indx[-1]
        pick.append(i)
        xx1 = np.maximum(x1[i], x1[indx])
        yy1 = np.maximum(y1[i], y1[indx])
        xx2 = np.minimum(x2[i], x2[indx])
        yy2 = np.minimum(y2[i], y2[indx])

        w = np.maximum(0.0, xx2 - xx1 + 1.0)
        h = np.maximum(0.0, yy2 - yy1 + 1.0)

        inter = w * h
        o = inter / (area[i] + area[indx] - inter)

        indx = indx[o <= overlap_thres]

    return np.asarray(pick)


def boxes_filter(boxes, scores, config_proposal):
    if len(scores.shape) == 1:
        aboxes = np.hstack((boxes, scores[:, np.newaxis]))
    else:
        aboxes = np.hstack((boxes, scores))

    if config_proposal.per_nms_topN > 0:
        aboxes = aboxes[:min(len(aboxes), config_proposal.per_nms_topN)]

    if config_proposal.nms_overlap_thres > 0 and config_proposal.nms_overlap_thres < 1:
        aboxes = aboxes[nms(aboxes, config_proposal.nms_overlap_thres)]

    if config_proposal.after_nms_topN > 0:
        aboxes = aboxes[:min(len(aboxes), config_proposal.after_nms_topN)]

    return aboxes[:, 0:4]


def nms_filter(boxes, scores, config_detection):
    """
    boxes: N x 80
    scores: N x 20
    """
    pred_boxes = []
    pred_scores = []
    labels = []
    for i in range(scores.shape[1]):
        aboxes = np.hstack((boxes[:, i * 4: (i + 1) * 4], scores[:, i][:, np.newaxis]))
        aboxes = aboxes[nms(aboxes, config_detection.nms_overlap_thres)]
        if np.any(aboxes):
            pred_boxes.append(aboxes[:, :4])
            pred_scores.append(aboxes[:, 4])
            labels.append(np.ones(len(aboxes), dtype=int) * i)

    pred_boxes, pred_scores, pred_labels = np.concatenate(pred_boxes), np.concatenate(pred_scores), np.concatenate(labels)
    indx = pred_scores > config_detection.min_score
    return pred_boxes[indx], pred_scores[indx], pred_labels[indx]
