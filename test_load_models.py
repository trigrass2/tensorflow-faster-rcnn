from proposal_test_model import load_proposal_model
from detection_test_model import load_detection_model
from caffe_utils import extract_caffe_weights

import tensorflow as tf
import numpy as np

try:
    proposal_weights = np.load('models/vgg16_proposal_model.npy')
    detection_weights = np.load('models/vgg16_detection_model.npy')

except:
    proposal_weights = extract_caffe_weights('pretrained_models/proposal_test.prototxt',
                                             'pretrained_models/proposal_final',
                                             'models/vgg16_proposal_model.npy')

    detection_weights = extract_caffe_weights('pretrained_models/detection_test.prototxt',
                                              'pretrained_models/detection_final',
                                              'models/vgg16_detection_model.npy')

proposal_layers = load_proposal_model(proposal_weights)
detection_layers = load_detection_model(detection_weights)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
