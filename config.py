from collections import namedtuple
import numpy as np

classes = [
    'aeroplane'
    'bicycle'
    'bird'
    'boat'
    'bottle'
    'bus'
    'car'
    'cat'
    'chair'
    'cow'
    'diningtable'
    'dog'
    'horse'
    'motorbike'
    'person'
    'pottedplant'
    'sheep'
    'sofa'
    'train'
    'tvmonitor'
]

image_means = [103.9390, 116.7790, 123.6800]  # BGR order

anchors = np.asarray(
    [[-83, -39, 100, 56],
     [-175, -87, 192, 104],
     [-359, -183, 376, 200],
     [-55, -55, 72, 72],
     [-119, -119, 136, 136],
     [-247, -247, 264, 264],
     [-35, -79, 52, 96],
     [-79, -167, 96, 184],
     [-167, -343, 184, 360]])

Proposal = namedtuple('Proposal', ['drop_boxes_runoff_image',
                                   'image_means',
                                   'max_size',
                                   'min_box_size',
                                   'per_nms_topN',
                                   'nms_overlap_thres',
                                   'after_nms_topN',
                                   'scales',
                                   'anchors',
                                   'feat_stride'])

Detection = namedtuple('Detection', ['max_size',
                                     'nms',
                                     'scales'])


def faster_rcnn_voc0712_vgg():
    Config = namedtuple('Config', ['proposal', 'detection'])

    proposal = Proposal(True, image_means, 1000, 16, 6000, 0.7, 300, 600, anchors, 16)
    detection = Detection(1000, 0.3, 600)
    config = Config(proposal, detection)

    return config
