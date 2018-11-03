"""
实时对图片或者视频内的物体进行识别，并输出识别结果图片或录像
多线程实现基于生产者消费者模型
"""
import os
import sys
import datetime
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2

# 多线程
from threading import Thread
from queue import Queue

# 项目根目录
# ROOT_DIR = os.path.abspath("../../") # 运行时使用
ROOT_DIR = os.path.abspath("")        # VS code调试时使用
sys.path.append(ROOT_DIR)  # To find local version of the library

# Mask-RCNN 模型导入(实时版本)
from mrcnn import real_time_visual
import mrcnn.model as modellib
from mrcnn.model import log
# Import Mask RCNN
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images

# Grasba 数据集模型
from samples.grasba import grasba

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# 输出结果目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

# Path to Grasba trained weights
GRASBA_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_grasba_181019.h5")

# Grasba 设置文件
config = grasba.GrasbaConfig()
GRASBA_DIR = os.path.join(ROOT_DIR, "datasets/grasba")


# 视频捕捉线程
class Producer(Thread):
    # 获取 OpenCV 格式视频图像
    # cap: OpenCV 视频引用可为摄像头或者视频
    # q： 队列
    def __init__(self, q, cap):
        super(Producer, self).__init__()
        self.q = q
        self.cap = cap

    def run(self):
        while not self.q.full():
            success, temp_frame = self.cap.read()  # 读取一帧的图像
            if success:
                self.q.put(temp_frame)
                cv2.imshow('Real-time Capture', temp_frame)  # 输出实时图像
            if cv2.waitKey(1) & 0xFF == ord('q'):  # q 键退出
                break
        else:
            print("QUEUE FULL!!!")  # 队列满时退出


# Override the training configurations with a few
# changes for inferencing.
class InferenceConfig(config.__class__):
    # Run detection on one image at a time
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


config = InferenceConfig()
# config.display() 显示 config 的内容

# Device to load the neural network on.
# Useful if you're training a model on the same
# machine, in which case use CPU and leave the
# GPU for training.
# os.environ["CUDA_VISIBLE_DEVICES"] = "0" # comment if use gpu
DEVICE = "/gpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


# 执行函数
def apply_effect(image, result, colors):
    # 参数提取
    p = result[0]
    figsize = (image.shape[1]/100.0, image.shape[0]/100.0)
    boxes = p['rois']
    masks = p['masks']
    class_ids = p['class_ids']
    class_names = ['BG', 'ball', 'hand']
    scores = p['scores']

    output = real_time_visual.display_instances(image, boxes, masks, class_ids, class_names, scores,
                                                colors=colors, real_time=True, figsize=figsize)

    return output


if __name__ == '__main__':
    # 建立 inference 用模型
    with tf.device(DEVICE):
        model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR, config=config)

    # 加载预训练模型
    # Set path to grasba weights file
    weights_path = GRASBA_WEIGHTS_PATH
    print("Loading weights ", weights_path)
    model.load_weights(weights_path, by_name=True)

    # 标注颜色选取
    MAX_CLASSES = 5
    colors = real_time_visual.random_colors(MAX_CLASSES)

    # 加载摄像头
    vcapture = cv2.VideoCapture(0)
    ret, frame = vcapture.read()
    q = Queue(maxsize=1000)

    if ret:
        width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = vcapture.get(cv2.CAP_PROP_FPS)

        # 录像文件
        file_name = os.path.join(OUTPUT_DIR, "output_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        vwriter = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
        count = 0

        video_thread = Producer(q, vcapture)
        video_thread.start()
        
        while True:
            # 时刻准备接收帧
            if not q.empty():
                # 如果有帧在队列时
                image = q.get()
                # BGR -> RGB
                image = image[..., ::-1]
                # Detect objects
                result = model.detect([image], verbose=0)
                # 操作
                image_output = apply_effect(image, result, colors)
                # RGB -> BGR to save image to video
                image_output = image_output[..., ::-1]
                # Add image to video writer
                cv2.imshow('Real-time Detection', image_output)
                vwriter.write(image_output)
                count += 1
                if cv2.waitKey(1) & 0xFF == ord('q'):  # q键退出循环
                    break
            else:
                if cv2.waitKey(1) & 0xFF == ord('q'):  # q键退出循环
                    break
        video_thread.join()
        vwriter.release()
        cv2.destroyAllWindows()
        print("Saved to ", file_name)
    else:
        print('sorry, no camera. I quit\n')
