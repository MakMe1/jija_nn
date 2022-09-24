import cv2
import numpy as np
from math import sqrt
import time
import os
from random import randint


def crop(img: np.array, mask: np.array, size=224):
    res = []
    res_mask = []
    h, w = mask.shape
    for h_iter in range(0, h, size):
        if h - h_iter < size:
            h_iter = h - size
        for w_iter in range(0, w, size):
            if w - w_iter < size:
                w_iter = w - size
            res.append(img[h_iter:h_iter + size, w_iter:w_iter + size])
            res_mask.append(mask[h_iter:h_iter + size, w_iter:w_iter + size])
    return res, res_mask

img = cv2.imread("test_0.JPG")
img_mask = cv2.imread("test_0_mask.JPG", cv2.IMREAD_GRAYSCALE)
res, res_mask = crop(img, img_mask, 224)

print(f"{len(res)=}  {len(res_mask)=}")

index = 725
cv2.imshow("res", res[index])
cv2.imshow("res_mask", res_mask[index])
cv2.waitKey(0)

