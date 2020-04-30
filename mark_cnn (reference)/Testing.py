import time

from Networks import Model
from DataProperties import DataProperties
import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.keras.preprocessing import image
import tensorflow as tf
import cv2


def testing(model):
    data = DataProperties()
    unet = model.load()

    path = 'data/testing/images'

    image_list = os.listdir(path)
    image_list.sort()
    idx = range(1, len(image_list))

    alpha = 1.
    for i in idx:
        st_img = time.time()
        x = image.load_img(f'{path}/{image_list[i]}', target_size=(data.height, data.height))
        x = image.img_to_array(x) / 255.

        z = unet.predict(np.expand_dims(x, axis=0))
        z = np.squeeze(z)

        # z[z < 0.4] = 0.0
        # z[z > 0.4] = 1.0

        cv2.imwrite(f'outputs/predicted/pred_{image_list[i]}', z * 255)

        # fig = plt.figure(figsize=(20, 20))
        # ax = fig.add_subplot(1, 2, 1)
        # ax.set_title(f'Image')
        # ax.imshow(x)
        #
        # # result = x.copy()
        # # x[z == 1] = [0, 0, 1]
        # # result = cv2.addWeighted(x, alpha, result, 1 - alpha, 0, result)
        # result = z
        # ax3 = fig.add_subplot(1, 2, 2)
        # ax3.set_title(f'Generated_Mask')
        # ax3.imshow(result)
        # fig.savefig(f'outputs/comparison/output_{index*i}.png')
        # plt.close()
        print(time.time() - st_img)

    return


def check_size_original_data():
    data = DataProperties()

    path = 'data/original'

    image_list = os.listdir(f'{path}/images')
    mask_list = os.listdir(f'{path}/masks')
    image_list.sort()
    mask_list.sort()
    idx = range(1, len(image_list))

    alpha = 0.5
    for i in idx:
        im = image.load_img(f'{path}/images/{image_list[i]}', target_size=(data.height, data.height))
        im = image.img_to_array(im) / 255.

        mask = image.load_img(f'{path}/masks/{mask_list[i]}', target_size=(data.height, data.height))
        mask = image.img_to_array(mask) / 255.

        im = cv2.addWeighted(mask, alpha, im, 1 - alpha, 0, im)
        fig = plt.figure(figsize=(20, 20))
        plt.imshow(im)
        fig.savefig(f'outputs/check/{image_list[i]}')
        plt.close()


testing(Model('preTrained'))
# check_size_original_data()
