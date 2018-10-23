######################################################
# 实时对图片或者视频内的物体进行识别，并输出识别结果图片或录像
######################################################

import os
import sys
import random
import math
import re
import time
import datetime
import numpy as np
import skimage
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
import colorsys
import matplotlib.patches as patches

# Root directory of the project
ROOT_DIR = os.path.abspath("../../")
# ROOT_DIR = os.path.abspath("")  # 调试时使用

# Import Mask RCNN
sys.path.append(ROOT_DIR)  # To find local version of the library
from mrcnn import utils
from mrcnn import visualize
from mrcnn.visualize import display_images

# 实时可视化
from mrcnn import real_time_visual
# from mrcnn.real_time_visual import display_instances, random_colors
import mrcnn.model as modellib
from mrcnn.model import log

# Grasba 模型
from samples.grasba import grasba

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# 输出目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "results")

# Path to Grasba trained weights
GRASBA_WEIGHTS_PATH = os.path.join(ROOT_DIR, "mask_rcnn_grasba_181019.h5")

# Grasba 设置文件
config = grasba.GrasbaConfig()
GRASBA_DIR = os.path.join(ROOT_DIR, "datasets/grasba")


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
os.environ["CUDA_VISIBLE_DEVICES"] = "0" # comment if use gpu
DEVICE = "/cpu:0"  # /cpu:0 or /gpu:0

# Inspect the model in training or inference modes
# values: 'inference' or 'training'
# TODO: code for 'training' test mode not ready yet
TEST_MODE = "inference"


def get_ax(rows=1, cols=1, size=16):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.

    Adjust the size attribute to control how big to render images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax

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


# 建立 inference 用模型
with tf.device(DEVICE):
    model = modellib.MaskRCNN(mode="inference", model_dir=MODEL_DIR,
                              config=config)

# 加载预训练模型
# Set path to grasba weights file
weights_path = GRASBA_WEIGHTS_PATH
# Or, load the last model you trained
# weights_path = model.find_last()
# Load weights
print("Loading weights ", weights_path)
model.load_weights(weights_path, by_name=True)

# 加载数据
# 图像和视频选择
image_path = os.path.join(GRASBA_DIR, "val", "red_ball.png")
# image_path = None
video_path = os.path.join(GRASBA_DIR, "val", "red_ball.mp4")
video_path = None
# 标注颜色选取
MAX_CLASSES = 5
colors = real_time_visual.random_colors(MAX_CLASSES)

if image_path:
    print("Running on {}".format(image_path))
    # 读取图片
    image = skimage.io.imread(image_path)
    # 识别物体
    result = model.detect([image], verbose=1)
    # 操作
    image_output = apply_effect(image, result, colors)
    # 输出
    file_name = os.path.join(OUTPUT_DIR, "output_{:%Y%m%dT%H%M%S}.png".format(datetime.datetime.now()))
    skimage.io.imsave(file_name, image_output)
elif video_path:
    import cv2
    vcapture = cv2.VideoCapture(video_path)
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # video writer
    file_name = os.path.join(OUTPUT_DIR, "output_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    vwriter = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
    count = 0
    success = True
    while success:
        print("fname: ", count)
        success, image = vcapture.read()
        if success:
            # OpenCV returns images as BGR, convert to RGB
            image = image[..., ::-1]
            # Detect objects
            result = model.detect([image], verbose=0)
            # 操作
            image_output = apply_effect(image, result, colors)
            # RGB -> BGR to save image to video
            image_output = image_output[..., ::-1]
            #print(image_output.shape)
            # Add image to video writer
            vwriter.write(image_output)
            count += 1
    vwriter.release()
print("Saved to ", file_name)
