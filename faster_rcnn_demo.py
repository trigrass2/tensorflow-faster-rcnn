from preprocessing import process_image
from proposal_test_model import load_proposal_model, proposal_test_model
from detection_test_model import load_detection_model, roi_pooling_layer, detection_test_model
from caffe_utils import extract_caffe_weights
from fast_rcnn_utils import filter_boxes
from nms import nms
from tools import plot_image_with_bbox

import cv2
import numpy as np
import namedtuple
import tensorflow as tf

image_name_list = ['1.jpg']

config = namedtuple('proposal',
                    'detection',
                    'nms')

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

for name in image_name_list:
    im = cv2.imread(name)[:, :, ::-1]
    im_prec = process_image(im, config.image_process)
    boxes, scores, feats = proposal_test_model(sess, im_prec, proposal_layers, config.proposal)
    boxes, scores = filter_boxes(boxes, scores, config.nms_before)

    roi_output = roi_pooling_layer(feats, boxes)

    boxes, scores = detection_test_model(sess, roi_output, detection_layers, config.detection)
    boxes, scores = nms(boxes, scores, config.nms_final)

    plot_image_with_bbox(im, boxes, scores, config.classes)
