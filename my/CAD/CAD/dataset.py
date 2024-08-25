import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import numpy as np
import pandas as pd
import time


def Unfold(max_x, max_y, winH, winW, stepW, stepH):
    x_points_l = np.array([i for i in range(0, max_x, stepW) if i + winH <= max_x])
    y_points_u = np.array([i for i in range(0, max_y, stepH) if i + winW <= max_y])

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


def rotate(bbox, angles, cell_num):
    r_matrix = np.hstack((np.cos(angles), -np.sin(angles), np.sin(angles), np.cos(angles), np.cos(angles),
                          -np.sin(angles), np.sin(angles), np.cos(angles), np.cos(angles), -np.sin(angles),
                          np.sin(angles), np.cos(angles), np.cos(angles), -np.sin(angles), np.sin(angles),
                          np.cos(angles)))
    r_matrix = r_matrix.reshape(cell_num, 8, 2).transpose(2, 0, 1).reshape(2, -1)

    # 旋转后的坐标
    rotated_bbox = r_matrix * bbox
    rotated_bbox = np.sum(rotated_bbox, axis=0)
    rotated_bbox = rotated_bbox.reshape(cell_num, 8)

    return rotated_bbox


def Affine_cell(file1, file2, cell_p):
    ps = pd.read_csv(f'{file1}/{file2}/affine_points.csv')
    src_p = np.array([ps.iloc[0, :2], ps.iloc[0, 2:4], ps.iloc[0, 4:6], ps.iloc[0, 6:8], ps.iloc[0, 8:10]])
    dst_p1 = np.array([ps.iloc[1, :2], ps.iloc[1, 2:4], ps.iloc[1, 4:6], ps.iloc[1, 6:8], ps.iloc[1, 8:10]])
    dst_p2 = np.array([ps.iloc[2, :2], ps.iloc[2, 2:4], ps.iloc[2, 4:6], ps.iloc[2, 6:8], ps.iloc[2, 8:10]])
    M1, _ = cv2.estimateAffinePartial2D(src_p, dst_p1)
    M2, _ = cv2.estimateAffinePartial2D(src_p, dst_p2)
    affine_cell_p1 = np.dot(M1, cell_p.T)
    affine_cell_p1 = affine_cell_p1.T
    affine_cell_p2 = np.dot(M2, cell_p.T)
    affine_cell_p2 = affine_cell_p2.T
    affine_cell_p = np.vstack((affine_cell_p1, affine_cell_p2))
    return affine_cell_p


def make_bbox(affine_cell_p, rectangle):
    bbox_x, bbox_y = affine_cell_p[:, 0] - rectangle[:, 0] / 2, affine_cell_p[:, 1] - rectangle[:, 1] / 2
    bbox_x = bbox_x.reshape(-1, 1)
    bbox_y = bbox_y.reshape(-1, 1)
    bbox_lu = np.hstack((bbox_x, bbox_y))
    bbox_ld = bbox_lu.copy()
    bbox_ld[:, 1] += rectangle[:, 1]
    bbox_ru = bbox_lu.copy()
    bbox_ru[:, 0] += rectangle[:, 0]
    bbox_rd = bbox_ld.copy()
    bbox_rd[:, 0] += rectangle[:, 0]
    bbox = np.hstack((bbox_lu, bbox_ru, bbox_rd, bbox_ld))
    return bbox


def find_windows(cell_num, bboxs, windows):
    point_mask = np.zeros((cell_num, windows.shape[0]), dtype=bool)
    for m in range(cell_num):
        mask_lu_x = np.logical_and(bboxs[m, :, 0].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bboxs[m, :, 0].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_lu_y = np.logical_and(bboxs[m, :, 1].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bboxs[m, :, 1].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_lu = np.logical_and(mask_lu_x, mask_lu_y)
        mask_ru_x = np.logical_and(bboxs[m, :, 2].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bboxs[m, :, 2].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_ru_y = np.logical_and(bboxs[m, :, 3].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bboxs[m, :, 3].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_ru = np.logical_and(mask_ru_x, mask_ru_y)
        mask_rd_x = np.logical_and(bboxs[m, :, 4].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bboxs[m, :, 4].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_rd_y = np.logical_and(bboxs[m, :, 5].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bboxs[m, :, 5].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_rd = np.logical_and(mask_rd_x, mask_rd_y)
        mask_ld_x = np.logical_and(bboxs[m, :, 6].reshape(-1, 1) >= windows[:, 0].reshape(-1, 1),
                                   bboxs[m, :, 6].reshape(-1, 1) <= windows[:, 2].reshape(-1, 1))
        mask_ld_y = np.logical_and(bboxs[m, :, 7].reshape(-1, 1) >= windows[:, 1].reshape(-1, 1),
                                   bboxs[m, :, 7].reshape(-1, 1) <= windows[:, 5].reshape(-1, 1))
        mask_ld = np.logical_and(mask_ld_x, mask_ld_y)
        mask_others_x = np.logical_and(bboxs[m, :, 0].reshape(-1, 1) <= windows[:, 0].reshape(-1, 1),
                                       bboxs[m, :, 2].reshape(-1, 1) >= windows[:, 2].reshape(-1, 1))
        mask_others_y = np.logical_and(bboxs[m, :, 1].reshape(-1, 1) <= windows[:, 1].reshape(-1, 1),
                                        bboxs[m, :, 5].reshape(-1, 1) >= windows[:, 5].reshape(-1, 1))
        mask_others = np.logical_or(mask_others_x, mask_others_y)
        mask = np.logical_or.reduce((mask_lu, mask_ru, mask_rd, mask_ld, mask_others))
        point_mask[m] = np.squeeze(mask)
    point = [np.where(point_mask[m])[0] for m in range(point_mask.shape[0])]
    return point


def draw_bbox(img, i, j, bboxs, big_bboxs, windows):
    top_left = np.clip(((bboxs[i, j, :2] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    bottom_right = np.clip(((bboxs[i, j, 4:6] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    top_right = np.clip(((bboxs[i, j, 2:4] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    bottom_left = np.clip(((bboxs[i, j, 6:8] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    big_top_left = np.clip(((big_bboxs[i, :2] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    big_bottom_right = np.clip(((big_bboxs[i, 4:6] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    big_top_right = np.clip(((big_bboxs[i, 2:4] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    big_bottom_left = np.clip(((big_bboxs[i, 6:8] - windows[j, 0:2]) * 0.25).astype(int), 0, 1024)
    img = cv2.line(img, top_left, top_right, (0, 0, 255), 2, 4)
    img = cv2.line(img, top_right, bottom_right, (0, 0, 255), 2, 4)
    img = cv2.line(img, bottom_right, bottom_left, (0, 0, 255), 2, 4)
    img = cv2.line(img, bottom_left, top_left, (0, 0, 255), 2, 4)
    img = cv2.line(img, big_top_left, big_top_right, (0, 0, 255), 2, 4)
    img = cv2.line(img, big_top_right, big_bottom_right, (0, 0, 255), 2, 4)
    img = cv2.line(img, big_bottom_right, big_bottom_left, (0, 0, 255), 2, 4)
    img = cv2.line(img, big_bottom_left, big_top_left, (0, 0, 255), 2, 4)
    return img


if __name__ == '__main__':
    file1_name = ['PWYL', 'PWYN-C', 'PWYN-S', 'PWZF-C', 'PWZF-S']
    file2_name = ['CS', 'SS']
    for file1 in file1_name:
        for file2 in file2_name:
            img_file = pd.read_csv(f'{file1}/{file2}/img_win.csv')
            max_x, max_y, winH, winW, stepW, stepH = img_file.iloc[0]
            df = pd.read_csv(f'{file1}/{file2}/{file2}_position.csv').dropna()
            coordinates = df.iloc[:, [4, 5]]
            cell_p = coordinates.to_numpy()
            cell_p = np.hstack((cell_p, np.ones((cell_p.shape[0], 1))))
            affine_cell_p = Affine_cell(file1, file2, cell_p)
            cell_num = affine_cell_p.shape[0]

            # 得到边界框的尺寸
            bbox = df.iloc[:, 10]
            bbox = bbox.to_numpy()
            bbox = np.array([list(map(int, rect.split('*'))) for rect in bbox])
            bbox = np.vstack((bbox, bbox))

            # 得到边界框顶点坐标
            bbox_p = make_bbox(affine_cell_p, bbox)

            c_p = np.hstack((affine_cell_p, affine_cell_p, affine_cell_p, affine_cell_p))
            bbox_p -= c_p
            new_bbox_p = bbox_p.reshape(-1, 2).T
            new_bbox_p = np.repeat(new_bbox_p, 2, axis=1)
            bbox_p += c_p
            bbox_p = bbox_p.reshape(cell_num, 4, 2)

            # 得到旋转角度
            angles = df.iloc[:, 6]
            angles = angles.to_numpy().reshape(-1, 1)
            angles = np.vstack((angles, angles))
            angles = np.radians(angles)

            rotated_bbox = rotate(new_bbox_p, angles, cell_num)
            rotated_bbox += c_p

            # 盖住旋转框的水平框
            rect_p_l_x = np.min(rotated_bbox[:, (0, 2, 4, 6)], axis=1).reshape(-1, 1)
            rect_p_r_x = np.max(rotated_bbox[:, (0, 2, 4, 6)], axis=1).reshape(-1, 1)
            rect_p_y_u = np.min(rotated_bbox[:, (1, 3, 5, 7)], axis=1).reshape(-1, 1)
            rect_p_y_d = np.max(rotated_bbox[:, (1, 3, 5, 7)], axis=1).reshape(-1, 1)
            big_bbox = np.hstack((rect_p_l_x, rect_p_y_u, rect_p_r_x, rect_p_y_u, rect_p_r_x, rect_p_y_d, rect_p_l_x, rect_p_y_d))

            # 生成滑动窗口
            windows = Unfold(max_x, max_y, winH, winW, stepW, stepH)
            bboxs = np.repeat(rotated_bbox[:, np.newaxis, :], windows.shape[0], axis=1)
            point = find_windows(cell_num, bboxs, windows)

            out_dir = f'datasets/train_bboxes'
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)

            for i in range(cell_num):
                for j in point[i]:
                    img = cv2.imread(f'{out_dir}/{file1}_{file2}_{j}.jpg')
                    img = draw_bbox(img, i, j, bboxs, big_bbox, windows)
                    cv2.imwrite(f'{out_dir}/{file1}_{file2}_{j}.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 70])
            print(f'{file1}_{file2} finished')