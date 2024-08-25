import cv2
import torch
import numpy as np
from ultralytics.nn.autobackend import AutoBackend
import os


def preprocess_warpAffine(image, dst_width=1024, dst_height=1024):
    scale = min((dst_width / image.shape[1], dst_height / image.shape[0]))
    ox = (dst_width - scale * image.shape[1]) / 2
    oy = (dst_height - scale * image.shape[0]) / 2
    M = np.array([
        [scale, 0, ox],
        [0, scale, oy]
    ], dtype=np.float32)

    img_pre = cv2.warpAffine(image, M, (dst_width, dst_height), flags=cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_CONSTANT, borderValue=(114, 114, 114))
    IM = cv2.invertAffineTransform(M)

    img_pre = (img_pre[:, :, ::-1] / 255.0).astype(np.float32)
    img_pre = img_pre.transpose(2, 0, 1)[None]
    img_pre = torch.from_numpy(img_pre)
    return img_pre, IM


def postprocess(pred, IM=[], conf_thres=0.1):
    boxes = []
    for item in pred[0]:
        cx, cy, w, h = item[:4]
        label = item[4:].argmax()
        confidence = item[4 + label]
        if confidence < conf_thres:
            continue
        left = cx - w * 0.5
        top = cy - h * 0.5
        right = cx + w * 0.5
        bottom = cy + h * 0.5
        boxes.append([left, top, right, bottom, confidence, label])

    boxes = np.array(boxes)
    try:
        lr = boxes[:, [0, 2]]
    except:
        return
    tb = boxes[:, [1, 3]]
    boxes[:, [0, 2]] = IM[0][0] * lr + IM[0][2]
    boxes[:, [1, 3]] = IM[1][1] * tb + IM[1][2]

    return boxes


def hsv2bgr(h, s, v):
    h_i = int(h * 6)
    f = h * 6 - h_i
    p = v * (1 - s)
    q = v * (1 - f * s)
    t = v * (1 - (1 - f) * s)

    r, g, b = 0, 0, 0

    if h_i == 0:
        r, g, b = v, t, p
    elif h_i == 1:
        r, g, b = q, v, p
    elif h_i == 2:
        r, g, b = p, v, t
    elif h_i == 3:
        r, g, b = p, q, v
    elif h_i == 4:
        r, g, b = t, p, v
    elif h_i == 5:
        r, g, b = v, p, q

    return int(b * 255), int(g * 255), int(r * 255)


def random_color(id):
    h_plane = (((id << 2) ^ 0x937151) % 100) / 100.0
    s_plane = (((id << 3) ^ 0x315793) % 100) / 100.0
    return hsv2bgr(h_plane, s_plane, 1)


def affine_boxes(boxes, i):
    src_img = cv2.imread(f'my/CAD/CAD/PWYS/{file2}/PWYS-{file2}.jpg')
    h, w = src_img.shape[:2]
    w_num = (w - 4096) // 2048 + 1
    x = i % w_num
    y = i // w_num
    x_t = x * 2048
    y_t = y * 2048
    boxes *= 4
    boxes[:, [0, 2]] += x_t
    boxes[:, [1, 3]] += y_t
    return boxes


def bbox_iou(box1, box2,  eps=1e-9):
    # Returns the IoU of box1 to box2. box1 is 4, box2 is nx4
    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
    b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]

    # Intersection area
    inter = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clamp(0) * \
            (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clamp(0)

    # Union Area
    w1, h1 = b1_x2 - b1_x1, b1_y2 - b1_y1 + eps
    w2, h2 = b2_x2 - b2_x1, b2_y2 - b2_y1 + eps
    union = w1 * h1 + w2 * h2 - inter + eps

    iou = inter / union

    return iou  # IoU


def my_nms(bboxes, iou_threshold=0.2):
    mask = np.ones(len(bboxes), dtype=bool)
    index = 0
    while True in mask[index+1:]:
        max_con_bbox = bboxes[index]
        max_con_bbox = max_con_bbox[np.newaxis, :]
        max_con_bbox = np.repeat(max_con_bbox, len(bboxes), axis=0)
        ious = bbox_iou(max_con_bbox, bboxes)
        mask = np.logical_and(mask, ious <= iou_threshold)
        index = np.where(mask[index+1:] == True)[0][0] + index + 1
    return bboxes[mask]


if __name__ == "__main__":
    for file2 in ['CS', 'SS']:
        pre_root = f'datasets/Data/images/test/{file2}'
        file_list = os.listdir(pre_root)
        pre_model_path = 'weights/yolov10b_best.pt'
        out_dir = 'datasets/Data/images/prediction_big'
        all_win_bboxes = []
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        for file in file_list:
            img = cv2.imread(os.path.join(pre_root, file))

            img_pre, IM = preprocess_warpAffine(img)

            model = AutoBackend(weights=pre_model_path)
            names = model.names
            result = model(img_pre)['one2one'][0].transpose(-1, -2)
            boxes = postprocess(result, IM)
            if boxes is None:
                continue
            boxes.sort(key=lambda x: x[4], reverse=True)
            boxes_np = np.array(boxes)
            boxes_af = affine_boxes(boxes_np, int(file.split('.')[0].rsplit('_', 1)[-1]))
            all_win_bboxes.append(boxes_af)
            # for obj in boxes_af:
            #     left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            #     confidence = obj[4]
            #     label = int(obj[5])
            #     color = random_color(label)
            #     cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
            #     caption = f"{names[label]} {confidence:.2f}"
            #     w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
            #     cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            #     cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
            # cv2.putText(img, str(len(boxes)), (0, 50), 0, 1, (130, 130, 130), 3)
        all_win_bboxes = np.array(all_win_bboxes)
        img_bboxes = my_nms(all_win_bboxes)
        for obj in img_bboxes:
            left, top, right, bottom = int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3])
            confidence = obj[4]
            label = int(obj[5])
            color = random_color(label)
            cv2.rectangle(img, (left, top), (right, bottom), color=color, thickness=2, lineType=cv2.LINE_AA)
            caption = f"{names[label]} {confidence:.2f}"
            w, h = cv2.getTextSize(caption, 0, 1, 2)[0]
            cv2.rectangle(img, (left - 3, top - 33), (left + w + 10, top), color, -1)
            cv2.putText(img, caption, (left, top - 5), 0, 1, (0, 0, 0), 2, 16)
        cv2.imwrite(os.path.join(out_dir, f'PWYS_{file2}.jpg'), img)