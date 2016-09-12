import tensorflow as tf


def var(weight):
    return tf.Variable(weight, tf.float32)


def fc(inp, w, b, act='relu'):
    var_w = var(w)
    var_b = var(b)
    out = tf.nn.wxs(inp, var_w, var_b)
    if act == 'relu':
        return tf.nn.relu(out)
    elif act == 'linear':
        return out
    elif act == 'softmax':
        return tf.nn.softmax(act)
    else:
        raise NotImplementedError


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
