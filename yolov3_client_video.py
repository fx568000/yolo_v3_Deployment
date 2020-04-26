import tensorflow as tf
from PIL import Image
import numpy as np
import argparse
import base64
import scipy
import grpc
import time
import cv2
from utils import load_coco_names,draw_boxes,get_boxes_and_inputs,get_boxes_and_inputs_pb,non_max_suppression,load_graph,letter_box_image
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2

# 参数配置
parser = argparse.ArgumentParser(description="YOLO-V3 test procedure.")
parser.add_argument("--server", type=str,default="10.10.32.179:8500", # 10.10.32.179:8500
                    help="PredictionService host:port(gRPC).")
# parser.add_argument("--rstp_server", type=str,default="rtsp://admin:fx123456@192.168.43.171/Streaming/Channels/1", #
#                     help="rstp_server:rstp video monitor.")
# parser.add_argument("--input_flag", type=str,default="video", # image,video,rstp
#                     help="input flag:image,video,rstp......")
parser.add_argument("--input_video", type=str,default="./data/test.avi", # 20191219.mp4 test.avi
                    help="The path of the input video.")
# parser.add_argument("--input_img", type=str,default="./data/messi.jpg", #
#                     help="Input image.")
parser.add_argument("--temp_img", type=str,default="./data/temp.jpg", #
                    help="temp image.")
# parser.add_argument("--output_img", type=str,default="./output/Result.jpg", #
#                     help="Output  image.")
parser.add_argument("--class_names", type=str,default="./data/coco.names", #./data/coco.names
                    help="File with class names.")
parser.add_argument("--data_format", type=str,default="NHWC", #
                    help="Data format: NCHW (gpu only) / NHWC.")
parser.add_argument("--size", type=int,default="416", #
                    help="Image size.")
parser.add_argument("--conf_threshold", type=float,default="0.5", #
                    help="Confidence threshold.")
parser.add_argument("--iou_threshold", type=float,default="0.4", #
                    help="IoU threshold.")
parser.add_argument("--gpu_memory_fraction", type=float,default="0.5", #
                    help="Gpu memory fraction to use.")
args = parser.parse_args()

# 从硬盘读取图片并转换成可传输的字节流
def base64_encode_img(filename):
    with open(filename, 'rb') as file:
        jpeg_bytes = file.read()
    jpeg_b64_str = base64.b64encode(jpeg_bytes).decode('utf-8')
    return jpeg_b64_str, jpeg_bytes


def main(argv=None):
    # GPU配置
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=args.gpu_memory_fraction)
    # config = tf.ConfigProto(gpu_options=gpu_options,log_device_placement=False,)
    # 类别、视频或图像输入
    classes = load_coco_names(args.class_names)

    vid = cv2.VideoCapture(args.input_video)
    video_frame_cnt = int(vid.get(7))  # AVI:10148  RSTP: 中无总帧数属性 视频文件中的帧数
    timeF = 10  # 分帧率 130ms配合2
    fpsnum = int(vid.get(1))  # 基于以0开始的被捕获或解码的帧索引
    if (fpsnum % timeF == 0):
        for i in range(video_frame_cnt):
            ret, img_ori = vid.read()
            # 图像填充
            img_ori = cv2.cvtColor(img_ori, cv2.COLOR_BGR2RGB)
            img_ori = Image.fromarray(img_ori) # CV2图片转PIL
            img_resized = letter_box_image(img_ori,img_ori.size[1], img_ori.size[0], args.size, args.size, 128)
            img_resized = img_resized.astype(np.float32)
            # 图像插值
            # img = cv2.resize(img_ori, (args.size, args.size))
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # cv2默认为bgr顺序
            # img_resized = np.asarray(img, np.float32)
            # 编码方式1
            # scipy.misc.imsave(args.temp_img, img_resized)
            # _, jpeg_bytes = base64_encode_img(args.temp_img)
            # 编码方式2
            img_encode = cv2.imencode('.jpg', img_resized)[1]
            data_encode = np.array(img_encode)
            jpeg_bytes = data_encode.tostring()
            start_time = time.time()
            # 服务器通讯配置
            channel = grpc.insecure_channel(args.server)
            stub = prediction_service_pb2.PredictionServiceStub(channel)
            request = predict_pb2.PredictRequest()
            request.model_spec.name = 'yolov3_2'
            request.model_spec.signature_name = 'predict_images'
            # 等待服务器答复
            request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(jpeg_bytes, shape=[1]))
            response = stub.Predict(request, 10.0)
            # 对返回值进行操作
            results = {}
            for key in response.outputs:
                tensor_proto = response.outputs[key]
                nd_array = tf.contrib.util.make_ndarray(tensor_proto)
                results[key] = nd_array
            detected_boxes = results['scores']
            # nms计算
            filtered_boxes = non_max_suppression(detected_boxes,confidence_threshold=args.conf_threshold,iou_threshold=args.iou_threshold)
            end_time = time.time()
            difference_time = end_time - start_time  # 网络运行时间
            # 画图
            draw_boxes(filtered_boxes, img_ori, classes, (args.size, args.size), True)
            # 输出图像
            cv2charimg = cv2.cvtColor(np.array(img_ori), cv2.COLOR_RGB2BGR) # PIL图片转cv2 图片
            cv2.putText(cv2charimg, '{:.2f}ms'.format((difference_time) * 1000), (40, 40), 0,
                        fontScale=1, color=(0, 255, 0), thickness=2)
            cv2.imshow('image', cv2charimg)
            if cv2.waitKey(1) & 0xFF == ord('q'): # 视频退出
                break


if __name__ == '__main__':
    tf.app.run()