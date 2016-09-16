from preprocessing import process_image
from proposal_test_model import load_proposal_model, proposal_test_model
from detection_test_model import load_detection_model, detection_test_model
from caffe_utils import extract_caffe_weights
from fast_rcnn_utils import generate_proposal_boxes, generate_detection_boxes
from nms import boxes_filter, nms_filter
from tools import plot_image_with_bbox
from config import faster_rcnn_voc0712_vgg, classes

import cv2
import numpy as np
import tensorflow as tf
from pylab import *

image_name_list = ['data/000005.jpg', 'data/001763.jpg', 'data/004545.jpg', 'data/000542.jpg', 'data/000456.jpg', 'data/001150.jpg']
config = faster_rcnn_voc0712_vgg()

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
    im = cv2.imread(name)
    im_prec, resize_scale = process_image(im, config.proposal)
    deltas, scores, feats = proposal_test_model(sess, im_prec, proposal_layers)
    pred_boxes, scores = generate_proposal_boxes(feats.shape[1:3], deltas, scores, im.shape[0:2], im_prec.shape[0:2], config.proposal)
    pred_boxes = boxes_filter(pred_boxes, scores, config.proposal)

    deltas, scores = detection_test_model(sess, feats, pred_boxes, detection_layers, resize_scale)
    boxes = generate_detection_boxes(deltas, pred_boxes, im.shape[0:2])
    pred_boxes, pred_scores, labels = nms_filter(boxes, scores, config.detection)

    plot_image_with_bbox(im[:, :, ::-1], pred_boxes, pred_scores, labels, classes)
    show()
