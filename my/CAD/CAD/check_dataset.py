import os

path_img = r'datasets\Image'
path_label = r'datasets\txt'


# 获取该目录下所有文件，存入列表中
fileList_img = os.listdir(path_img)
fileList_label = os.listdir(path_label)
print(len(fileList_img), len(fileList_label))
n = 0