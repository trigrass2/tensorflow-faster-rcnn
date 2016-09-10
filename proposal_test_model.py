import tensorflow as tf
import numpy as np


def var(weight):
    return tf.Variable(weight, tf.float32)


def conv(inp, w, b, relu=True):
    var_w = var(w)
    var_b = var(b)
    out = tf.nn.bias_add(tf.nn.conv2d(inp, var_w, [1, 1, 1, 1], 'SAME', True), var_b)

    if relu:
        return tf.nn.relu(out)
    else:
        return out


def pool2x2(inp):
    return tf.nn.max_pool(inp, [1, 2, 2, 1], [1, 2, 2, 1], 'SAME')


def load_proposal_model(ws):
    assert len(ws) == 32
    blob = tf.placeholder(tf.float32)

    conv1_1 = conv(blob, ws[0], ws[1])
    conv1_2 = conv(conv1_1, ws[2], ws[3])
    pool1 = pool2x2(conv1_2)

    conv2_1 = conv(pool1, ws[4], ws[5])
    conv2_2 = conv(conv2_1, ws[6], ws[7])
    pool2 = pool2x2(conv2_2)

    conv3_1 = conv(pool2, ws[8], ws[9])
    conv3_2 = conv(conv3_1, ws[10], ws[11])
    conv3_3 = conv(conv3_2, ws[12], ws[13])
    pool3 = pool2x2(conv3_3)

    conv4_1 = conv(pool3, ws[14], ws[15])
    conv4_2 = conv(conv4_1, ws[16], ws[17])
    conv4_3 = conv(conv4_2, ws[18], ws[19])
    pool4 = pool2x2(conv4_3)

    conv5_1 = conv(pool4, ws[20], ws[21])
    conv5_2 = conv(conv5_1, ws[22], ws[23])
    conv5_3 = conv(conv5_2, ws[24], ws[25])

    conv_proposal1 = conv(conv5_3, ws[26], ws[27])

    proposal_bbox_pred = conv(conv_proposal1, ws[30], ws[31], relu=False)

    proposal_cls_score = conv(conv_proposal1, ws[28], ws[29], relu=False)
    reshape_score = tf.reshape(proposal_cls_score, [-1, 2])
    proposal_cls_prob = tf.nn.softmax(reshape_score)

    return blob, proposal_bbox_pred, proposal_cls_prob, conv_proposal1


def proposal_test_model(sess, im_input, layers):
    blob, proposal_bbox_pred, proposal_cls_prob, conv_proposal1 = layers

    if len(im_input.shape) == 3:
        im_input = im_input[np.newaxis, :, :, :]

    return sess.run([proposal_bbox_pred, proposal_cls_prob, conv_proposal1],
                    feed_dict={blob: im_input})
