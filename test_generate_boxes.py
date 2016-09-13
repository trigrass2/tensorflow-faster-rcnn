from preprocessing import process_image
from proposal_test_model import load_proposal_model, proposal_test_model
from caffe_utils import extract_caffe_weights
from fast_rcnn_utils import generate_proposal_boxes
from config import faster_rcnn_voc0712_vgg
from nms import boxes_filter

import cv2
import numpy as np
import tensorflow as tf
import ipdb

image_name_list = ['data/000005.jpg']
config = faster_rcnn_voc0712_vgg()

gtbox = np.array(
    [[263.0, 211.0, 324.0, 339.0],
     [165.0, 264.0, 253.0, 372.0],
     [5.0, 244.0, 67.0, 374.0],
     [241.0, 194.0, 295.0, 299.0],
     [277.0, 186.0, 312.0, 220.0]])

gtcls = np.array([8, 8, 8, 8, 8])

try:
    proposal_weights = np.load('models/vgg16_proposal_model.npy')

except:
    proposal_weights = extract_caffe_weights('pretrained_models/proposal_test.prototxt',
                                             'pretrained_models/proposal_final',
                                             'models/vgg16_proposal_model.npy')

proposal_layers = load_proposal_model(proposal_weights)

sess = tf.Session()
sess.run(tf.initialize_all_variables())

for name in image_name_list:
    im = cv2.imread(name)[:, :, ::-1]
    im_prec = process_image(im, config.proposal)
    deltas, scores, feats = proposal_test_model(sess, im_prec, proposal_layers)
    # ipdb.set_trace()
    pred_boxes, scores = generate_proposal_boxes(feats.shape[1:3], deltas, scores, im.shape[0:2], im_prec.shape[0:2], config.proposal)
    pred_boxes = boxes_filter(pred_boxes, scores, config.proposal)

from tools import plot_image_with_bbox

plot_image_with_bbox(im, pred_boxes[:20])
