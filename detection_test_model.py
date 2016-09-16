import tensorflow as tf
from tf_utils import fc
import numpy as np
from roi_pooling_layer import roi_pooling_layer

def pos_round(x):
    return np.floor(x + 0.5)


def load_detection_model(ws):
    assert len(ws) == 8
    roi_blob = tf.placeholder(tf.float32, shape=[None, 7, 7, 512])
    flat_roi_blob = tf.reshape(roi_blob, [-1, 25088])
    fc6 = fc(flat_roi_blob, ws[0], ws[1])
    fc7 = fc(fc6, ws[2], ws[3])
    cls_prob = fc(fc7, ws[4], ws[5], 'softmax')
    bbox_pred = fc(fc7, ws[6], ws[7], 'linear')
    return roi_blob, cls_prob, bbox_pred

'''
# legacy code, wrong

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
        st_w, st_h, ed_w, ed_h = roi[i]
        ed_w = min(ed_w, data.shape[2] - 1)
        ed_h = min(ed_w, data.shape[1] - 1)
        cp_h = max(ed_h - st_h + 1, 1)
        cp_w = max(ed_w - st_w + 1, 1)

        pool_sz_h = int(round(cp_h / float(pooled_h)))
        pool_sz_w = int(round(cp_w / float(pooled_w)))

        if pool_sz_h > 0 and pool_sz_w > 0:

            real_cp_h = min(pool_sz_h * pooled_h, data.shape[1] - st_h - 1)
            real_cp_w = min(pool_sz_w * pooled_w, data.shape[2] - st_w - 1)

            cropped_feature = tf.slice(data, [0, st_h, st_w, 0], [1, real_cp_h, real_cp_w, -1])

            outlist.append(tf.nn.max_pool(cropped_feature,
                                          ksize=[1, pool_sz_h, pool_sz_w, 1],
                                          strides=[1, pool_sz_h, pool_sz_w, 1],
                                          padding='SAME'))
        else:
            outlist.append(tf.zeros(shape=(1, 7, 7, 512)))

    return tf.concat(0, outlist)
'''


def map_rois_to_feat_rois(boxes, im_scale, upsampling_scale=0.0625):
    """
    Args:
        boxes: N x 4
        im_scale: scalar
    """
    rescaled_rois = pos_round((boxes - 1) * im_scale) + 1
    return np.asarray(pos_round((rescaled_rois - 1) * upsampling_scale), dtype=int)  # to 0 based index in python


def detection_test_model(sess, feats, boxes, layers, im_scale, upsampling_scale=0.0625):

    roi_blob, cls_prob, bbox_pred = layers

    feat_boxes = map_rois_to_feat_rois(boxes, im_scale, upsampling_scale)

    feats = np.asarray(feats, dtype=np.float)

    roi_output = roi_pooling_layer(feats, feat_boxes)

    return sess.run([bbox_pred, cls_prob], feed_dict={roi_blob: roi_output})
