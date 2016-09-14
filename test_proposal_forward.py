from proposal_test_model import load_proposal_model, proposal_test_model
import numpy as np
from preprocessing import process_image
from config import faster_rcnn_voc0712_vgg
import tensorflow as tf
import cv2

config = faster_rcnn_voc0712_vgg()

proposal_weights = np.load('models/vgg16_proposal_model.npy')
proposal_layers = load_proposal_model(proposal_weights)
sess = tf.Session()
sess.run(tf.initialize_all_variables())

im = cv2.imread('data/000005.jpg')

im_prec = process_image(im, config.proposal)
boxes, scores, feats = proposal_test_model(sess, im_prec, proposal_layers)
