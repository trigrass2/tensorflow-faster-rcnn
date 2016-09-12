from detection_test_model import load_detection_model, detection_test_model
import numpy as np
from config import faster_rcnn_voc0712_vgg
import tensorflow as tf

config = faster_rcnn_voc0712_vgg()

detection_weights = np.load('models/vgg16_detection_model.npy')

detection_layers = load_detection_model(detection_weights)

sess = tf.Session()
sess.run(tf.initialize_all_variables())


feats = np.asarray(np.random.randn(1, 38, 50, 512), dtype=np.float32)
boxes = np.array(
    [[100, 100, 200, 200],
     [120, 120, 250, 250]],
    dtype=int)
boxes, scores = detection_test_model(sess, feats, boxes, detection_layers, 1.0 / 16)
