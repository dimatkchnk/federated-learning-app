import cv2
# from model import resize
from processing import show_images
import os
import tensorflow_datasets as tfds
# from model import *
import tensorflow as tf
from tensorflow.python.client import device_lib



image_path = "./10328_DDSM.png"
mask_path = "./10328_DDSM_mask.png"
#
image = cv2.imread(image_path)
mask = cv2.imread(mask_path)

print(image.shape)
print(mask.shape)
#
# resized_im, resized_mask = resize(image, mask)
#
# print(resized_im.shape)
# print(resized_mask.shape)
# show_images(resized_im, resized_mask)


# # Path to the current folder
# folder_path = f'/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/train/1'
#
# output_folder = f'/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/masks/train/1'
#
# # Iterate through each file in the folder
# for filename in os.listdir(folder_path)[:10]:
#     print(filename)
#
# dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
#
# print(dataset)
# train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
# test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

print(device_lib.list_local_devices())
