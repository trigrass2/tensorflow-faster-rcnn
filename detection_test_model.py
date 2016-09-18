import tensorflow as tf
from tf_utils import fc
import numpy as np
from roi_pooling_layer import roi_pooling_op


def pos_round(x):
    return np.floor(x + 0.5)


def load_detection_model(ws):
    assert len(ws) == 8
    data_blob = tf.placeholder(tf.float32)
    rois_blob = tf.placeholder(tf.float32, shape=[None, 5])
    pool5, _ = roi_pooling_op.roi_pool(data_blob, rois_blob, 7, 7, 0.0625)
    flat_pool5 = tf.reshape(pool5, [-1, 25088])
    fc6 = fc(flat_pool5, ws[0], ws[1])
    fc7 = fc(fc6, ws[2], ws[3])
    cls_prob = fc(fc7, ws[4], ws[5], 'softmax')
    bbox_pred = fc(fc7, ws[6], ws[7], 'linear')
    return data_blob, rois_blob, cls_prob, bbox_pred


def map_rois_to_feat_rois(boxes, im_scale):
    """
    Args:
        boxes: N x 4
        im_scale: scalar
    """
    return pos_round((boxes - 1) * im_scale) + 1


def detection_test_model(sess, feats, boxes, layers, im_scale):

    data_blob, rois_blob, cls_prob, bbox_pred = layers

    feat_boxes = map_rois_to_feat_rois(boxes, im_scale)

    feat_boxes = np.hstack((np.zeros((len(feat_boxes), 1)), feat_boxes))  # add batch_index

    pred_boxes, pred_scores = sess.run([bbox_pred, cls_prob],
                                       feed_dict={data_blob: feats,
                                       rois_blob: feat_boxes})

    return pred_boxes[:, 4:], pred_scores[:, 1:]  # delete background
