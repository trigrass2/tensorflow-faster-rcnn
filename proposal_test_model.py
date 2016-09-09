from keras.layers import Convolution2D, MaxPooling2D


def load_proposal_model(model_npy):
    assert len(model_npy) == 8
    conv1_1 = Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv1_1').set_weights(model_npy[0:2])
    conv1_2 = Convolution2D(64, 3, 3, border_mode='same', activation='relu', name='conv1_2').set_weights(model_npy[2:4])
    pool1 = MaxPooling2D((2, 2), 2, border_mode='same', name='pool1')

    conv2_1 = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv2_1').set_weights(model_npy[4:6])
    conv2_2 = Convolution2D(128, 3, 3, border_mode='same', activation='relu', name='conv2_2').set_weights(model_npy[6:8])
    pool2 = MaxPooling2D((2, 2), 2, border_mode='same', name='pool2')

    conv3_1 = Convolution2D(256, 3, 3, border_mode='same', activation='relu', name='conv3_1').set_weights(model_npy[8:10])
    conv3_2 = Convolution2D(256, 3, 3, border_mode='same', activation='relu', name='conv3_2').set_weights(model_npy[10:12])
    conv3_3 = Convolution2D(256, 3, 3, border_mode='same', activation='relu', name='conv3_3').set_weights(model_npy[12:14])
    pool3 = MaxPooling2D((2, 2), 2, border_mode='same', name='pool3')

    conv4_1 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv4_1').set_weights(model_npy[14:16])
    conv4_2 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv4_2').set_weights(model_npy[16:18])
    conv4_3 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv4_3').set_weights(model_npy[18:20])
    pool4 = MaxPooling2D((2, 2), 2, border_mode='same', name='pool4')

    conv5_1 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv5_1').set_weights(model_npy[20:22])
    conv5_2 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv5_2').set_weights(model_npy[22:24])
    conv5_3 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv5_3').set_weights(model_npy[24:28])

    conv_proposal1 = Convolution2D(512, 3, 3, border_mode='same', activation='relu', name='conv_proposal1').set_weights(model_npy[28:30])
    proposal_bbox_pred = Convolution2D(18, 1, 1, border_mode='same', name='proposal_bbox_pred').set_weights(model_npy[30:32])
    proposal_cls_prob = Convolution2D(36, 1, 1, border_mode='same', activation='softmax', name='proposal_cls_prob').set_weights(model_npy[32:34])

    return (conv1_1, conv1_2, pool1,
            conv2_1, conv2_2, pool2,
            conv3_1, conv3_2, conv3_3, pool3,
            conv4_1, conv4_2, conv4_3, pool4,
            conv5_1, conv5_2, conv5_3,
            conv_proposal1, proposal_bbox_pred, proposal_cls_prob)


def proposal_test_model(sess, im_input, layers):
    blob = im_input
    for i in range(18):
        blob = layers[i](blob)

    proposal_bbox_pred_output = layers[-2](blob)
    proposal_cls_prob_output = layers[-1](blob)

    return sess.run([proposal_bbox_pred_output, proposal_cls_prob_output, blob])
