# -*- coding: utf-8 -*-

import numpy as np
import tensorflow as tf
import yolo_v3
import yolo_v3_tiny
from tensorflow.python import pywrap_tensorflow
from PIL import Image, ImageDraw

from utils import load_weights, load_coco_names, detections_boxes, freeze_graph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'class_names', 'data/coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'models/yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NCHW', 'Data format: NCHW (gpu only) / NHWC')
tf.app.flags.DEFINE_string(
    'output_graph', 'models/pb/frozen_darknet_yolov3_model.pb', 'Frozen tensorflow protobuf model output path')

tf.app.flags.DEFINE_bool(
    'tiny', False, 'Use tiny version of YOLOv3')
tf.app.flags.DEFINE_bool(
    'spp', False, 'Use SPP version of YOLOv3')
tf.app.flags.DEFINE_integer(
    'size', 416, 'Image size')

checkpoint_path = 'models/yolov3.ckpt'

def main(argv=None):
    if FLAGS.tiny:
        model = yolo_v3_tiny.yolo_v3_tiny
    elif FLAGS.spp:
        model = yolo_v3.yolo_v3_spp
    else:
        model = yolo_v3.yolo_v3

    classes = load_coco_names(FLAGS.class_names)

    # placeholder for detector inputs
    inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], "inputs")

    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format) # 得到yolov3整体模型（包含模型输出(?, 10647, (num_classes + 5))）
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

    # Sets the output nodes in the current session
    boxes = detections_boxes(detections) # 1，将整体输出分解为box结果与概率数值结果；2、将结果名称定义为output_boxes再放入graph中

    with tf.Session() as sess:
        sess.run(load_ops)
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor_name: ", key)
        freeze_graph(sess, FLAGS.output_graph)

if __name__ == '__main__':
    tf.app.run()
