import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time

img = cv2.imread('PWYL/CS/PWYL-CS.jpg')

# 获取图片形状
height, width, channels = img.shape

# 定义窗口大小和步幅
winH, winW = 4096, 4096
stepH, stepW = 2048, 2048
# 计算步幅
strides = (stepH * img.strides[0], stepW*img.strides[1], img.strides[0], img.strides[1], img.strides[2])

# 计算输出形状
out_shape = ((height - winH) // stepH + 1, (width - winW) // stepW + 1, winH, winW, channels)

# 使用 np.lib.stride_tricks.as_strided 生成滑动窗口
s = time.perf_counter()
windows2 = np.lib.stride_tricks.as_strided(img, shape=out_shape, strides=strides)
windows2 = windows2.reshape(-1, winH, winW, channels)
for i in range(windows2.shape[0]):
    cv2.imwrite(f'PWYL/CS/slice_1/PWYL_CS_{i}.jpg', windows2[i])
e = time.perf_counter()
print(f'run time: {(e - s)*1000:.2f}ms')