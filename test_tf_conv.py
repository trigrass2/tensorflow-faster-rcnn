import numpy as np
from tf_utils import conv
from preprocessing import process_image
from config import faster_rcnn_voc0712_vgg
import tensorflow as tf
import cv2

config = faster_rcnn_voc0712_vgg()

ws = np.load('models/vgg16_proposal_model.npy')
w = ws[0]
b = ws[1]

inp = cv2.imread('data/000005.jpg')
inp = process_image(inp, config.proposal)
inp = np.asarray(inp[np.newaxis, :, :, :], dtype=np.float32)
c = conv(inp, w, b)

sess = tf.Session()
sess.run(tf.initialize_all_variables())
