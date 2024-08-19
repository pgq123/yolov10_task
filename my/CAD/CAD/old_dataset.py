import os

# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import pandas as pd
import time


def find_windows(cell_num, bbox_repeated, windows):
    point_mask = np.zeros((cell_num, windows.shape[0]), dtype=bool)
    for m in range(cell_num):
        mask_lu_x = np.logical_and(bbox_repeated[m, :, 0].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bbox_repeated[m, :, 0].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_lu_y = np.logical_and(bbox_repeated[m, :, 1].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bbox_repeated[m, :, 1].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_lu = np.logical_and(mask_lu_x, mask_lu_y)
        mask_ru_x = np.logical_and(bbox_repeated[m, :, 2].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bbox_repeated[m, :, 2].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_ru_y = np.logical_and(bbox_repeated[m, :, 3].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bbox_repeated[m, :, 3].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_ru = np.logical_and(mask_ru_x, mask_ru_y)
        mask_rd_x = np.logical_and(bbox_repeated[m, :, 4].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bbox_repeated[m, :, 4].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_rd_y = np.logical_and(bbox_repeated[m, :, 5].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bbox_repeated[m, :, 5].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_rd = np.logical_and(mask_rd_x, mask_rd_y)
        mask_ld_x = np.logical_and(bbox_repeated[m, :, 6].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bbox_repeated[m, :, 6].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_ld_y = np.logical_and(bbox_repeated[m, :, 7].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bbox_repeated[m, :, 7].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_ld = np.logical_and(mask_ld_x, mask_ld_y)
        mask_others_x = np.logical_and(bbox_repeated[m, :, 0].reshape(-1, 1) <= windows[:, 0].reshape(-1, 1),
                                       bbox_repeated[m, :, 2].reshape(-1, 1) >= windows[:, 2].reshape(-1, 1))
        mask_others_y = np.logical_and(bbox_repeated[m, :, 1].reshape(-1, 1) <= windows[:, 1].reshape(-1, 1),
                                        bbox_repeated[m, :, 5].reshape(-1, 1) >= windows[:, 5].reshape(-1, 1))
        mask_others = np.logical_or(mask_others_x, mask_others_y)
        mask = np.logical_or.reduce((mask_lu, mask_ru, mask_rd, mask_ld, mask_others))
        point_mask[m] = np.squeeze(mask)
    point = [np.where(point_mask[m])[0] for m in range(point_mask.shape[0])]
    return point


if __name__ == '__main__':
    src_p = np.array([[50.69, 53.785], [50.19, 145.285], [309.339, 54.485], [321.689, 162.785],
                      [44.104, 128.251], [49.883, 167.297], [46.386, 52.047], [241.417, 59.209],
                      [326.975, 52.403], [326.321, 128.09]])
    dst_p1 = np.array([[3966, 18447], [3860, 28968], [33690, 18674], [35051, 31127], [3176, 27016],
                       [3810, 31493], [3427, 18221], [25886, 19177], [35695, 18368], [35642, 27165]])

    m1, inliers = cv2.estimateAffinePartial2D(src_p, dst_p1, method=cv2.RANSAC, ransacReprojThreshold=50)

    df = pd.read_csv('PWYL/CS/PWYL_position_size.csv').dropna()
    coordinates = df.iloc[:, [4, 5]]
    cell_p = coordinates.to_numpy()
    # num:312 index:271
    cell_p = np.hstack((cell_p, np.ones((cell_p.shape[0], 1))))

    # 得到所有点的坐标
    affine_cell_p = np.dot(m1, cell_p.T)
    affine_cell_p = affine_cell_p.T

    # 得到边界框的尺寸
    rectangle = df.iloc[:, 10]
    rectangle = rectangle.to_numpy()
    rectangle = np.array([list(map(int, rect.split('*'))) for rect in rectangle])

    # 得到边界框顶点坐标
    rect_p_x, rect_p_y = affine_cell_p[:, 0] - rectangle[:, 0] / 2, affine_cell_p[:, 1] - rectangle[:, 1] / 2
    rect_p_x = rect_p_x.reshape(-1, 1)
    rect_p_y = rect_p_y.reshape(-1, 1)
    rect_p_lu = np.hstack((rect_p_x, rect_p_y))
    rect_p_ld = np.hstack((rect_p_x, rect_p_y + rectangle[:, 1].reshape(-1, 1)))

    rect_p_x, rect_p_y = affine_cell_p[:, 0] + rectangle[:, 0] / 2, affine_cell_p[:, 1] + rectangle[:, 1] / 2
    rect_p_x = rect_p_x.reshape(-1, 1)
    rect_p_y = rect_p_y.reshape(-1, 1)
    rect_p_rd = np.hstack((rect_p_x, rect_p_y))
    rect_p_ru = np.hstack((rect_p_x, rect_p_y - rectangle[:, 1].reshape(-1, 1)))
    rect_p = np.hstack((rect_p_lu, rect_p_ru, rect_p_rd, rect_p_ld))
    c_p = np.hstack((affine_cell_p, affine_cell_p, affine_cell_p, affine_cell_p))
    rect_p -= c_p
    n_rect_p = rect_p.reshape(-1, 2).T
    n_rect_p = np.repeat(n_rect_p, 2, axis=1)
    rect_p += c_p
    rect_p = rect_p.reshape(1190, 4, 2)

    # 得到旋转角度
    angles = df.iloc[:, 6]
    angles = angles.to_numpy()
    angles = np.radians(angles)

    r_matrix = np.vstack((np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles), np.cos(angles),
                          -np.sin(angles), np.sin(angles), np.cos(angles), np.cos(angles), -np.sin(angles),
                          np.sin(angles), np.cos(angles), np.cos(angles), -np.sin(angles), np.sin(angles),
                          np.cos(angles))).T
    r_matrix = r_matrix.reshape(1190, 8, 2).transpose(2, 0, 1).reshape(2, -1)

    # 旋转后的坐标
    rotated_rect_p = r_matrix * n_rect_p
    rotated_rect_p = np.sum(rotated_rect_p, axis=0)
    rotated_rect_p = rotated_rect_p.reshape(1190, 8)
    rotated_rect_p += c_p

    # 盖住旋转框的水平框
    rect_p_l_x = np.min(rotated_rect_p[:, (0, 2, 4, 6)], axis=1).reshape(-1, 1)
    rect_p_r_x = np.max(rotated_rect_p[:, (0, 2, 4, 6)], axis=1).reshape(-1, 1)
    rect_p_y_u = np.min(rotated_rect_p[:, (1, 3, 5, 7)], axis=1).reshape(-1, 1)
    rect_p_y_d = np.max(rotated_rect_p[:, (1, 3, 5, 7)], axis=1).reshape(-1, 1)
    big_rect_p = np.hstack((rect_p_l_x, rect_p_y_u, rect_p_r_x, rect_p_y_u, rect_p_r_x, rect_p_y_d, rect_p_l_x, rect_p_y_d))

    # 定义窗口大小和步长
    step = 512
    max_x, max_y = 37262, 32763

    # 生成所有可能的窗口起始点
    x_points_l = np.array([i for i in range(0, max_x, step) if i + 1024 <= max_x])
    y_points_u = np.array([i for i in range(0, max_y, step) if i + 1024 <= max_y])

    w_lu = np.repeat([x_points_l], 62, axis=0)
    w_lu = np.repeat(w_lu, 2, axis=1)
    w_lu[:, 1::2] = np.repeat(y_points_u[:, np.newaxis], 71, axis=1)
    w_lu = w_lu.reshape(4402, 2)

    w_ru = np.copy(w_lu)
    w_ru[:, 0] += 1024

    w_rd = np.copy(w_ru)
    w_rd[:, 1] += 1024

    w_ld = np.copy(w_rd)
    w_ld[:, 0] -= 1024

    windows = np.hstack((w_lu, w_ru, w_rd, w_ld))
    windows = windows.reshape(4402, 8)
    rect_p_repeated = np.repeat(rotated_rect_p[:, np.newaxis, :], 4402, axis=1)

    s = time.perf_counter()
    point = find_windows(1190, rect_p_repeated, windows)
    e = time.perf_counter()
    print(e - s)

    # for i in range(1190):
    #     for j in point[i][0]:
    #         img = cv2.imread(f'PWYL/CS/slice/{j}.jpg')
    #         top_left = np.clip(rect_p_repeated[i, j, :2].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         bottom_right = np.clip(rect_p_repeated[i, j, 4:6].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         top_right = np.clip(rect_p_repeated[i, j, 2:4].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         bottom_left = np.clip(rect_p_repeated[i, j, 6:8].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         big_top_left = np.clip(big_rect_p[i, :2].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         big_bottom_right = np.clip(big_rect_p[i, 4:6].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         big_top_right = np.clip(big_rect_p[i, 2:4].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         big_bottom_left = np.clip(big_rect_p[i, 6:8].astype(int) - windows[j, 0:2].astype(int), 0, 1024)
    #         img = cv2.line(img, top_left, top_right, (0, 0, 255), 10, 4)
    #         img = cv2.line(img, top_right, bottom_right, (0, 0, 255), 10, 4)
    #         img = cv2.line(img, bottom_right, bottom_left, (0, 0, 255), 10, 4)
    #         img = cv2.line(img, bottom_left, top_left, (0, 0, 255), 10, 4)
    #         img = cv2.line(img, big_top_left, big_top_right, (0, 0, 255), 10, 4)
    #         img = cv2.line(img, big_top_right, big_bottom_right, (0, 0, 255), 10, 4)
    #         img = cv2.line(img, big_bottom_right, big_bottom_left, (0, 0, 255), 10, 4)
    #         img = cv2.line(img, big_bottom_left, big_top_left, (0, 0, 255), 10, 4)
    #         try:
    #             cv2.imwrite(f'PWYL/CS/slice/{j}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
    #         except:
    #             continue
    #         e = time.perf_counter()