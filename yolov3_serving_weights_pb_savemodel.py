# -*- coding: utf-8 -*-

import tensorflow as tf
import yolo_v3
import yolo_v3_tiny
from tensorflow.python import pywrap_tensorflow


from utils import load_weights, load_coco_names, detections_boxes, freeze_graph

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string(
    'class_names', 'data/coco.names', 'File with class names')
tf.app.flags.DEFINE_string(
    'weights_file', 'models/yolov3.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string(
    'data_format', 'NHWC', 'Data format: NCHW (gpu only) / NHWC')
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

    # 定义网络对外的接口, 服务器端是自动base64解码，所以拿到的数据已经解码过了
    # 但还需将byte格式转换为图像矩阵格式
    jpeg_vec_bytes = tf.placeholder(tf.string, shape=None, name=None)
    jpeg_sca_bytes = tf.reshape(jpeg_vec_bytes, [])  #
    jpeg_ndarr = tf.image.decode_jpeg(jpeg_sca_bytes, fancy_upscaling=False) # 从字符串变为数组，且将标量形式字节流解码成图片，！！！这里参数必须设置成False否则不能与客服端的结果匹配
    jpeg_ndarr = tf.image.resize_images(jpeg_ndarr,  [FLAGS.size, FLAGS.size], method=0)    # 将图片拉伸成希望的尺寸
    inputs = tf.reshape(jpeg_ndarr, [1, FLAGS.size, FLAGS.size, 3], "inputs")
    # placeholder for detector inputs 原址输入参量处
    # inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3], "inputs")
    # 加载yolov3模型
    with tf.variable_scope('detector'):
        detections = model(inputs, len(classes), data_format=FLAGS.data_format) # 得到yolov3整体模型（包含模型输出(?, 10647, (num_classes + 5))）
        load_ops = load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)
    # Sets the output nodes in the current session
    boxes = detections_boxes(detections) # 1，将整体输出分解为box结果与概率数值结果；2、将结果名称定义为output_boxes再放入graph中
    # checkpoint读取
    with tf.Session() as sess:
        sess.run(load_ops)
        reader = pywrap_tensorflow.NewCheckpointReader(checkpoint_path)
        var_to_shape_map = reader.get_variable_to_shape_map()
        for key in var_to_shape_map:
            print("tensor_name: ", key)
        #############################################
        # output_node_names = ["output_boxes","inputs",]
        # output_node_names = ",".join(output_node_names)
        #
        # output_graph_def = tf.graph_util.convert_variables_to_constants(
        #     sess,tf.get_default_graph().as_graph_def(),output_node_names.split(","))
        #
        # with tf.gfile.GFile(FLAGS.output_graph, "wb") as f:
        #     f.write(output_graph_def.SerializeToString())
        # print("{} ops written to {}.".format(len(output_graph_def.node), FLAGS.output_graph))
        #############################################
        # pb_savemodel模式存储
        export_path = 'models/pb/20191226'
        builder = tf.saved_model.builder.SavedModelBuilder(export_path)

        images = tf.saved_model.utils.build_tensor_info(jpeg_vec_bytes)
        boxes = tf.saved_model.utils.build_tensor_info(boxes)
        prediction_signature = (
            tf.saved_model.signature_def_utils.build_signature_def(
                inputs={'images': images},
                outputs={'scores': boxes},
                method_name=tf.saved_model.signature_constants.PREDICT_METHOD_NAME))
        builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'predict_images': prediction_signature},
            main_op=tf.tables_initializer(),
            strip_default_attrs=True)
        builder.save()

if __name__ == '__main__':
    tf.app.run()
