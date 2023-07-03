import tensorflow as tf
from keras import layers
import matplotlib.pyplot as plt
import numpy as np
import os
from tqdm import tqdm
import cv2
from sklearn.model_selection import train_test_split


def resize(input_image, input_mask):
    input_image = tf.image.resize(input_image, (128, 128), method="nearest")
    input_mask = tf.image.resize(input_mask, (128, 128), method="nearest")
    return input_image, input_mask


def augment(input_image, input_mask):
    if tf.random.uniform(()) > 0.5:
        # Random flipping of the image and mask
        input_image = tf.image.flip_left_right(input_image)
        input_mask = tf.image.flip_left_right(input_mask)
    return input_image, input_mask


def normalize(input_image, input_mask):
    input_image = tf.cast(input_image, tf.float32) / 255.0
    input_mask -= 1
    return input_image, input_mask


def display(display_list):
    plt.figure(figsize=(15, 15))
    title = ["Input Image", "True Mask", "Predicted Mask"]
    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


# BUILDING MODEL
def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)
    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)
    return x


def build_unet_model():
    # inputs
    inputs = layers.Input(shape=(128, 128, 3))
    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)
    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)
    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)
    # outputs
    outputs = layers.Conv2D(3, 1, padding="same", activation="softmax")(u9)
    # unet model with Keras Functional API
    unet_model = tf.keras.Model(inputs, outputs, name="U-Net")
    return unet_model


unet_model = build_unet_model()

# unet_model.summary()

# TRAINING MODEL
folder_path = "/home/dimatkchnk/praca_dyplomowa/images/train-for-unet"
images_path = "/home/dimatkchnk/praca_dyplomowa/images/train-for-unet/images"
masks_path = "/home/dimatkchnk/praca_dyplomowa/images/train-for-unet/masks"

images = []
masks = []

for filename in tqdm(os.listdir(images_path), colour="blue"):
    # Load the image
    image_path = os.path.join(images_path, filename)
    image = cv2.imread(image_path)

    # Load the corresponding mask
    mask_filename = filename.replace(".png", "_mask.png")
    mask_path = os.path.join(masks_path, mask_filename)
    mask = cv2.imread(mask_path)

    image, mask = resize(image, mask)
    image, mask = normalize(image, mask)

    images.append(image)
    masks.append(mask)

# Convert the lists to numpy arrays
images = np.array(images)
masks = np.array(masks)


# print(images[23].shape)
# print(masks[23].shape)

train_images, test_images, train_masks, test_masks = train_test_split(images, masks, test_size=0.2, random_state=42)

unet_model.compile(optimizer=tf.keras.optimizers.Adam(), loss="categorical_crossentropy", metrics="accuracy")

unet_model.fit(train_images, train_masks, batch_size=16, epochs=20, validation_data=(test_images, test_masks))
evaluation = unet_model.evaluate(test_images, test_masks)

print("Evaluation Loss:", evaluation[0])
print("Evaluation Accuracy:", evaluation[1])
