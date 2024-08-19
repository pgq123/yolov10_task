import os
os.environ["OPENCV_IO_MAX_IMAGE_PIXELS"] = pow(2, 40).__str__()
import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import time


file1 = 'PWYL'
file2 = 'CS'
image = cv2.imread(f'{file1}/{file2}/{file1}-{file2}.jpg')
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
data = torch.from_numpy(image).permute(2, 0, 1)
(winW, winH) = (4096, 4096)
stepSize = (2048, 2048)
unfolded = data.unsqueeze(0).unfold(2, winW, stepSize[0]).unfold(3, winH, stepSize[1])
# windows = windows.permute(0, 2, 1).view(num_windows, channels, winW, winH).numpy()
out_dir = f'dataset/{file1}/{file2}'
if not os.path.exists(out_dir):
    os.makedirs(out_dir, exist_ok=True)

# windows = windows.numpy()
s = time.perf_counter()
window = windows[0, :, :, :].permute(1, 2, 0).numpy()
cv2.imwrite(f'{file1}_{file2}_{0}.jpg', window)
# to_pil = transforms.ToPILImage()
# window = to_pil(window)
# window.save(f'test.jpg')
# e = time.perf_counter()
# print(f'time2: {e - s}')
# torch.save(window, f'{out_dir}/{file1}_{file2}_{0}.pt')
