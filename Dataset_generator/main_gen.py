import cv2
import numpy as np
from math import sqrt
import time
import os
from random import randint

brightness_thr = 50
brightness_default = 100
look_ahead = 100

number_of_train_data = 10
number_of_test_data = 5
dataset_path = "test_dataset/"
data_path = "microdst/"

extra_scale = 4
start_train_offtet = 20
start_test_offtet = 10


test_path = dataset_path + "test/"
train_path = dataset_path + "train/"
train_mask_path = dataset_path + "train_mask/"

bears_path = data_path + "bears/"
arctic_wb_path = data_path + "arctic_with_bears/"
arctic_path = data_path + "arctic_no_bears/"

bear_names = os.listdir(bears_path + "bears")
arctic_names = os.listdir(arctic_path)

mean_len = sqrt(pow(52, 2) + pow(49.8181, 2)) * extra_scale

gl_start = time.time()
start = time.time()
for iteration in range(start_train_offtet, number_of_train_data + start_train_offtet):
    bears_number = randint(1, 5)
    bear_indexes = np.random.randint(0, len(bear_names) - 1, bears_number)
    arctic_index = randint(0, len(arctic_names) - 1)

    arctic = cv2.imread(arctic_path + arctic_names[arctic_index])
    arctic_h, arctic_w, arctic_d = arctic.shape
    arctic_mask = np.zeros((arctic_h, arctic_w), np.uint8)

    for bear_iterator in range (0, bears_number):
        bear = cv2.imread(bears_path + "bears/" + bear_names[bear_indexes[bear_iterator]])
        bear_mask = cv2.imread(bears_path + "masks/" + bear_names[bear_indexes[bear_iterator]], cv2.IMREAD_GRAYSCALE)

        bear_h, bear_w = bear_mask.shape

        bear_size = sqrt(pow(bear_h, 2) + pow(bear_w, 2))
        scale = mean_len / bear_size

        bear_h = int(bear_h * scale)
        bear_w = int(bear_w * scale)

        bear = cv2.resize(bear, (bear_w, bear_h), interpolation=cv2.INTER_AREA)
        bear_mask = cv2.resize(bear_mask, (bear_w, bear_h), interpolation=cv2.INTER_AREA)

        bear_pos = {'x': randint(0, arctic_w - bear_w), 'y': randint(0, arctic_h - bear_h)}

        area = {'xl': max(bear_pos['x'] - look_ahead, 0), 'yl': max(bear_pos['y'] - look_ahead, 0), \
                'xr': min(bear_pos['x'] + look_ahead + bear_w, arctic_w),
                'yr': min(bear_pos['y'] + look_ahead + bear_h, arctic_h)}

        arctic_tiny = arctic[area['yl']:area['yr'], area['xl']:area['xr']]
        arctic_tiny = cv2.cvtColor(arctic_tiny, cv2.COLOR_BGR2GRAY)
        arctic_brightness = np.average(arctic_tiny)
        arctic_brightness = arctic_brightness if (arctic_brightness >= brightness_thr) else brightness_default


        val = 0
        num = 0
        bear_gray = cv2.cvtColor(bear, cv2.COLOR_BGR2GRAY)
        for i in range(0, bear_h):
            for j in range(0, bear_w):
                if bear_mask[i, j] > 0:
                    val += bear_gray[i, j]
                    num += 1
        bear_brightness = val / num

        brightness_scale = arctic_brightness / bear_brightness

        for i in range(0, bear_h):
            for j in range(0, bear_w):
                if bear_mask[i, j] > 0:
                    arctic[bear_pos['y'] + i, bear_pos['x'] + j] = bear[i, j] * brightness_scale
                    arctic_mask[bear_pos['y'] + i, bear_pos['x'] + j] = 255
                else:
                    arctic_mask[bear_pos['y'] + i, bear_pos['x'] + j] = 0

    cv2.imwrite(train_path + str(iteration) + ".JPG", arctic)
    cv2.imwrite(train_mask_path + str(iteration) + ".JPG", arctic_mask)
    print(f"train {iteration} generated")

end = time.time()
print(f"train data generation time {end - start}")
start = time.time()

for iteration in range(start_test_offtet, number_of_test_data + start_test_offtet):
    bears_number = randint(1, 5)
    bear_indexes = np.random.randint(0, len(bear_names) - 1, bears_number)
    arctic_index = randint(0, len(arctic_names) - 1)

    arctic = cv2.imread(arctic_path + arctic_names[arctic_index])
    arctic_h, arctic_w, arctic_d = arctic.shape

    for bear_iterator in range (0, bears_number):
        bear = cv2.imread(bears_path + "bears/" + bear_names[bear_indexes[bear_iterator]])
        bear_mask = cv2.imread(bears_path + "masks/" + bear_names[bear_indexes[bear_iterator]], cv2.IMREAD_GRAYSCALE)

        bear_h, bear_w = bear_mask.shape

        bear_size = sqrt(pow(bear_h, 2) + pow(bear_w, 2))
        scale = mean_len / bear_size

        bear_h = int(bear_h * scale)
        bear_w = int(bear_w * scale)

        bear = cv2.resize(bear, (bear_w, bear_h), interpolation=cv2.INTER_AREA)
        bear_mask = cv2.resize(bear_mask, (bear_w, bear_h), interpolation=cv2.INTER_AREA)

        bear_pos = {'x': randint(0, arctic_w - bear_w), 'y': randint(0, arctic_h - bear_h)}

        area = {'xl': max(bear_pos['x'] - look_ahead, 0), 'yl': max(bear_pos['y'] - look_ahead, 0), \
                'xr': min(bear_pos['x'] + look_ahead + bear_w, arctic_w),
                'yr': min(bear_pos['y'] + look_ahead + bear_h, arctic_h)}

        arctic_tiny = arctic[area['yl']:area['yr'], area['xl']:area['xr']]
        arctic_tiny = cv2.cvtColor(arctic_tiny, cv2.COLOR_BGR2GRAY)
        arctic_brightness = np.average(arctic_tiny)
        arctic_brightness = arctic_brightness if (arctic_brightness >= brightness_thr) else brightness_default


        val = 0
        num = 0
        bear_gray = cv2.cvtColor(bear, cv2.COLOR_BGR2GRAY)
        for i in range(0, bear_h):
            for j in range(0, bear_w):
                if bear_mask[i, j] > 0:
                    val += bear_gray[i, j]
                    num += 1
        bear_brightness = val / num

        brightness_scale = arctic_brightness / bear_brightness

        for i in range(0, bear_h):
            for j in range(0, bear_w):
                if bear_mask[i, j] > 0:
                    arctic[bear_pos['y'] + i, bear_pos['x'] + j] = bear[i, j] * brightness_scale

    cv2.imwrite(test_path + str(iteration) + ".JPG", arctic)
    print(f"test {iteration} generated")

end = time.time()
print(f"test data generation time {end - start}")

gl_end = time.time()
print(f"Total generation time {gl_end - gl_start}")