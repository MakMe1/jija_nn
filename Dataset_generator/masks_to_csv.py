import cv2
import numpy as np
import csv
import os

dataset_path = "test_dataset/"
train_mask_path = dataset_path + "train_mask/"
arctic_path = dataset_path + "train/"
mask_names = os.listdir(train_mask_path)

min_area = 0.95 * 180 * 35
max_area = 1.05 * 180 * 35

f = open('test.csv', 'w', encoding='UTF8', newline='')
writer = csv.writer(f)

for mask_index in mask_names:
    # img = cv2.imread(arctic_path + mask_index)
    gray = cv2.imread(train_mask_path + mask_index, cv2.IMREAD_GRAYSCALE)
    thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    for c in contours:
        area = cv2.contourArea(c)
        x = (max(c[:, 0, 0]) + min(c[:, 0, 0])) // 2
        y = (max(c[:, 0, 1]) + min(c[:, 0, 1])) // 2
        writer.writerow([mask_index, x, y])

f.close()