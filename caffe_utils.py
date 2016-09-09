# based on  https://github.com/ppwwyyxx/tensorpack/blob/master/tensorpack/utils/loadcaffe.py

import caffe
import numpy as np


def extract_caffe_weights(net_file, weights_file, save_file_name):
    net = caffe.Net(net_file, weights_file, caffe.TEST)
    params = []
    layer_names = net._layer_names
    blob_names = net.blobs.keys()

    for layername, layer in zip(layer_names, net.layers):
        if layer.type == 'Convolution':
            params.append(layer.blobs[0].data.transpose(2, 3, 1, 0))
            params.append(layer.blobs[1].data)

        if layer.type == 'InnerProduct':
            prev_blob_name = blob_names[blob_names.index(layername) - 1]
            prev_data_shape = net.blobs[prev_blob_name].data.shape[1:]
            W = layer.blobs[0].data
            b = layer.blobs[1].data
            if len(prev_data_shape) == 3:
                W = W.reshape((-1,) + prev_data_shape).transpose(2, 3, 1, 0)
                W = W.reshape((np.prod(prev_data_shape), -1))
            else:
                W = W.transpose()

            params.append(W)
            params.append(b)

        # TODO: add BatchNorm params extraction

    np.save(save_file_name, params)
