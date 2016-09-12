import tensorflow as tf
from tf_utils import fc
import numpy as np


def load_detection_model(ws):
    assert len(ws) == 8
    roi_blob = tf.placeholder(tf.float32, shape=[None, 7, 7, 512])
    flat_roi_blob = tf.reshape(roi_blob, [-1, 25088])
    fc6 = fc(flat_roi_blob, ws[0], ws[1])
    fc7 = fc(fc6, ws[2], ws[3])
    cls_prob = fc(fc7, ws[4], ws[5], 'softmax')
    bbox_pred = fc(fc7, ws[6], ws[7], 'linear')
    return roi_blob, cls_prob, bbox_pred


def roi_pooling_layer(data, roi):
    """
    Args:
        data: tf.Variable 1 x H x W x C
        roi: bbox N x 4
    """
    # faster rcnn default setting
    pooled_h = 7
    pooled_w = 7

    roi_num = len(roi)
    outlist = []
    for i in xrange(roi_num):
        st_h, st_w, cp_h, cp_w = roi[i]

        pool_sz_h = cp_h // pooled_h
        pool_sz_w = cp_w // pooled_w

        real_cp_h = pooled_h * pool_sz_h
        real_cp_w = pooled_w * pool_sz_w

        cropped_feature = tf.slice(data, [0, st_h, st_w, 0], [1, real_cp_h, real_cp_w, -1])

        outlist.append(tf.nn.max_pool(cropped_feature,
                                      ksize=[1, pool_sz_h, pool_sz_w, 1],
                                      strides=[1, pool_sz_h, pool_sz_w, 1],
                                      padding='SAME'))
    return tf.concat(0, outlist)


def map_rois_to_feat_rois(boxes, scale):
    """
    Args:
        boxes: N x 4
        scale: scalar
    """
    return np.asarray((boxes - 1) * scale + 1, dtype=int)


def detection_test_model(sess, feats, boxes, layers, scale):

    roi_blob, cls_prob, bbox_pred = layers
    feat_boxes = map_rois_to_feat_rois(boxes, scale)
    roi_output = roi_pooling_layer(feats, feat_boxes)
    roi_blob_data = sess.run(roi_output)
    return sess.run([bbox_pred, cls_prob], feed_dict={roi_blob: roi_blob_data})
