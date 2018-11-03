"""
生成标注用视频
"""
import cv2
import os
import sys
import datetime

# 设定根目录
ROOT_DIR = os.path.abspath("../../")  # 运行时使用
# ROOT_DIR = os.path.abspath("")        # VS code调试时使用
sys.path.append(ROOT_DIR)  # To find local version of the library
# 创建输出结果目录
OUTPUT_DIR = os.path.join(ROOT_DIR, "results", "sample_{:%Y%m%dT%H%M%S}".format(datetime.datetime.now()))
if os.path.exists(OUTPUT_DIR):
    raise Exception("输出目录已存在")
else:
    os.makedirs(OUTPUT_DIR)

# 设置摄像头
vcapture = cv2.VideoCapture(0)  # 使用第0个摄像头
ret, frame = vcapture.read()  # 读取一帧图像测试
if ret:
    width = int(vcapture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(vcapture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = vcapture.get(cv2.CAP_PROP_FPS)

    # 设置录像文件
    file_name = os.path.join(OUTPUT_DIR, "sample_{:%Y%m%dT%H%M%S}.avi".format(datetime.datetime.now()))
    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    vwriter = cv2.VideoWriter(file_name, fourcc, fps, (width, height))
    success = True

    while success:
        success, frame = vcapture.read()  # 读取一帧的图像
        cv2.imshow('Real-time Capture', frame)
        vwriter.write(frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    print("Saved to ", file_name)
    vwriter.release()
    vcapture.release()  # 释放摄像头
    cv2.destroyAllWindows()
else:
    print("NO CAMERA!!")
