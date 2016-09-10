from keras.layers import Dense
import tensorflow as tf


def load_detection_model(model_npy):
    assert len(model_npy) == 8
    fc6 = Dense(4096, activation='relu', name='fc6', weights=model_npy[0:2])
    fc7 = Dense(4096, activation='relu', name='fc7', weights=model_npy[2:4])
    cls_prob = Dense(21, activation='softmax', name='cls_prob', weights=model_npy[4:6])
    bbox_pred = Dense(84, name='bbox_pred', weights=model_npy[6:8])
    return fc6, fc7, cls_prob, bbox_pred


def roi_pooling_layer(data, roi):
    """
    Args:
        data: tf.Variable N x H x W x C
        roi: bbox N x 1 x 1 x 1 x 1
    """
    # faster rcnn default setting
    pooled_h = 7
    pooled_w = 7

    roi_num = len(roi)
    outlist = []
    for i in xrange(roi_num):
        btch_i, st_h, st_w, cp_h, cp_w = roi[i]

        pool_sz_h = cp_h // pooled_h
        pool_sz_w = cp_w // pooled_w

        real_cp_h = pooled_h * pool_sz_h
        real_cp_w = pooled_w * pool_sz_w

        cropped_feature = tf.slice(data, [btch_i, st_h, st_w, 0], [1, real_cp_h, real_cp_w, -1])

        outlist.append(tf.nn.max_pool(cropped_feature,
                                      ksize=[1, pool_sz_h, pool_sz_w, 1],
                                      strides=[1, pool_sz_h, pool_sz_w, 1],
                                      padding='SAME'))
    return tf.concat(0, outlist)


def detection_test_model(sess, roi_output, layers):
    fc6, fc7, cls_prob, bbox_pred = layers
    fc6_output = fc6(roi_output)
    fc7_output = fc7(fc6_output)
    cls_prob_output = cls_prob(fc7_output)
    bbox_pred_output = bbox_pred(fc7_output)

    return sess.run([bbox_pred_output, cls_prob_output])
