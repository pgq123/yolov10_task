import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import time


def Unfold(max_x, max_y, winH, winW, stepSize):
    x_points_l = np.array([i for i in range(0, max_x, stepSize[0]) if i + winH <= max_x])
    y_points_u = np.array([i for i in range(0, max_y, stepSize[1]) if i + winW <= max_y])

    x_wins_cnt = len(x_points_l)
    y_wins_cnt = len(y_points_u)
    wins_cnt = x_wins_cnt * y_wins_cnt

    w_lu = np.repeat([x_points_l], y_wins_cnt, axis=0)
    w_lu = np.repeat(w_lu, 2, axis=1)
    w_lu[:, 1::2] = np.repeat(y_points_u[:, np.newaxis], x_wins_cnt, axis=1)
    w_lu = w_lu.reshape(wins_cnt, 2)

    w_ru = np.copy(w_lu)
    w_ru[:, 0] += winW

    w_rd = np.copy(w_ru)
    w_rd[:, 1] += winH

    w_ld = np.copy(w_rd)
    w_ld[:, 0] -= winW

    windows = np.hstack((w_lu, w_ru, w_rd, w_ld))
    return windows

if __name__ == '__main__':
    # 自定义滑动窗口的大小
    file1_name = ['PWYL', 'PWYN-C', 'PWYN-S', 'PWYS', 'PWZF-C', 'PWZF-S']
    file2_name = ['CS', 'SS']
    for file1 in file1_name:
        for file2 in file2_name:
            image = cv2.imread(f'{file1}/{file2}/{file1}-{file2}.jpg')
            windows = Unfold(image.shape[1], image.shape[0], 4096, 4096, (2048, 2048))
            if file1 in ['PWYL', 'PWYN-C', 'PWYN-S', 'PWZF-C'] or (file1 == 'PWZF-S' and file2 == 'CS'):
                out_dir = f'datasets/Image/train'
            elif file1 == 'PWZF-S' and file2 == 'SS':
                out_dir = f'datasets/Image/val'
            else:
                out_dir = f'datasets/Image/test'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            for i in range(windows.shape[0]):
                img = image[windows[i, 1]:windows[i, 5], windows[i, 0]:windows[i, 2], :]
                img = cv2.resize(img, (1024, 1024))
                cv2.imwrite(f'{out_dir}/{file1}_{file2}_{i}.jpg', img)
            print(f'{file1}-{file2} finished')
            del image