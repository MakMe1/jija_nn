import numpy as np, cv2
import csv
import os

from PIL import Image


def get_centers(name:str, mask: np.ndarray, writer):
    # mask = mask.astype(np.uint8)
    # mask *= 255

    mask = mask * 255
    # mask = Image.fromarray(mask)
    # mask.convert()
    # mask.convertTo(mask, cv2.CV_8U1C, 255.0)
    mask = np.uint8(mask)
    thresh = cv2.threshold(mask, 30, 255, cv2.THRESH_BINARY)[1]

    min_area = 0.95 * 180 * 35
    max_area = 1.05 * 180 * 35

    contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours[0] if len(contours) == 2 else contours[1]

    result = []
    for c in contours:
        area = cv2.contourArea(c)
        x = (max(c[:, 0, 0]) + min(c[:, 0, 0])) // 2
        y = (max(c[:, 0, 1]) + min(c[:, 0, 1])) // 2
        writer.writerow([name, x, y])
        # print(f'added line: {name=}, {x=}, {y=}')

    # return result

# dataset_path = "test_dataset/"
# train_mask_path = dataset_path + "train_mask/"
# arctic_path = dataset_path + "train/"
# mask_names = os.listdir(train_mask_path)
#
# # min_area = 0.95 * 180 * 35
# # max_area = 1.05 * 180 * 35
#
# for mask_index in mask_names:
#     # img = cv2.imread(arctic_path + mask_index)
#     gray = cv2.imread(train_mask_path + mask_index, cv2.IMREAD_GRAYSCALE)
#
#     # tmp = get_centers(gray)
#     # writer.writerow([mask_index, get_centers(gray)[:]])
#
#     # thresh = cv2.threshold(gray, 30, 255, cv2.THRESH_BINARY)[1]
#
#     # contours = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     # contours = contours[0] if len(contours) == 2 else contours[1]
#
#     # for c in contours:
#     #     area = cv2.contourArea(c)
#     #     x = (max(c[:, 0, 0]) + min(c[:, 0, 0])) // 2
#     #     y = (max(c[:, 0, 1]) + min(c[:, 0, 1])) // 2
#     #     writer.writerow([mask_index, x, y])
#
# f.close()