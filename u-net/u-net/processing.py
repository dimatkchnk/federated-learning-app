import matplotlib.pyplot as plt
import pydicom
import numpy as np
import cv2
from pydicom.pixel_data_handlers.util import apply_modality_lut, apply_voi_lut, apply_windowing
from skimage import color, data, restoration
from scipy.signal import wiener
from PIL import Image
import gdcm
import libjpeg


def convert_to_16_bit_png_file(img_data, file_path, debug=False):
    # img_data = img_data.astype(np.uint16)
    # array_buffer = img_data.tobytes()
    # if debug:
    #     print(len(array_buffer))
    #     print(img_data.T.shape)
    # img = Image.new("I", img_data.T.shape)
    # img.frombytes(array_buffer, 'raw', "I;16")
    # img.save(file_path)

    # OPTION WITH cv2 LIBRARY
    img_data = img_data.astype(np.uint16)
    scaled_img = (np.maximum(img_data, 0) / img_data.max()) * 255.0
    cv2.imwrite(file_path, scaled_img)


def image_segmentation(img, show_thres=False):
    _, img_thres = cv2.threshold(img, 0.1, 255, cv2.THRESH_BINARY)

    #     kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 13))
    #     opening = cv2.morphologyEx(img_thres, cv2.MORPH_OPEN, kernel)
    #     closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
    #     closing = closing.astype(np.uint8)

    kernel_erosion = cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11))
    erosion = cv2.erode(img_thres, kernel_erosion, iterations=4)

    kernel_dilate = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (50, 50))
    dilation = cv2.dilate(erosion, kernel_dilate)

    # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    # smoothed_image = cv2.morphologyEx(img_thres, cv2.MORPH_CLOSE, kernel)

    dilation = dilation.astype(np.uint8)
    # smoothed_image = smoothed_image.astype(np.uint8)

    if show_thres:
        plt.figure(figsize=(12, 6))
        plt.imshow(dilation, cmap='gray')
        plt.show()

    img_out = cv2.bitwise_and(img, img, mask=dilation)

    return img_out


def show_images(img1, img2, title1='', title2='', cmap='gray'):
    plt.figure(figsize=(12,6))
    plt.subplot(121)
    plt.title(title1)
    plt.imshow(img1, cmap=cmap)

    plt.subplot(122)
    plt.title(title2)
    plt.imshow(img2, cmap=cmap)

    plt.show()


def show_hist(img):

    img = img.astype(np.uint8)
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])

    plt.figure(figsize=(12,12))
    plt.title('Histogram')
    plt.xlabel('Pixel Value')
    plt.ylabel('Frequency')
    plt.plot(hist)
    plt.xlim([0, 256])
    plt.show()


def load_image_pydicom(img_path, voi_lut=False, modality_lut=False, windowing=False, return_ds=False):
    dataset = pydicom.dcmread(img_path)
    img = dataset.pixel_array
    if voi_lut:
        img = apply_voi_lut(img, dataset)
    if modality_lut:
        img = apply_modality_lut(img, dataset)
    if windowing:
        img = apply_windowing(img, dataset)
    if dataset.PhotometricInterpretation == "MONOCHROME1":
        img = np.amax(img) - img
    if return_ds:
        return img, dataset
    else:
        return img


def connected_components(img, show_components=False):

    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    img = img.astype(np.uint16)
    # img = img.astype(np.uint8)

    # img_thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    # _, img_thres = cv2.threshold(img, 5, 255, cv2.THRESH_BINARY)
    _, img_thres = cv2.threshold(img, 127, 255, cv2.THRESH_OTSU)
    # _, img_thres = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
    # img_thres = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # show_images(img, img_thres)

    img_thres = img_thres.astype(np.uint8)

    # Apply the Component analysis function
    output = cv2.connectedComponentsWithStats(img_thres,
                                              4,
                                              cv2.CV_32S)
    (totalLabels, labels, stats, centroids) = output

    # Initialize a new image to
    # store all the output components
    mask = np.zeros(img.shape, dtype="uint8")

    # loop over the number of unique connected component labels
    for i in range(1, totalLabels):

        #text = "examining component {}/{}".format(i + 1, totalLabels)
        #print("[INFO] {}".format(text))

        # extract the connected component statistics and centroid for
        # the current label
        #x = stats[i, cv2.CC_STAT_LEFT]
        #y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # print(w)
        # print(h)
        # print(area)

        keep_width = w > 500
        keep_height = h > 1000
        keep_area = area > 500000

        if show_components:
            img_copy = img.copy()

            showComponentMask = (labels == i).astype("uint8") * 255

            show_images(img_copy, showComponentMask)

        # ensure the connected component we are examining passes all
        # three tests
        if all((keep_width, keep_height, keep_area)):
            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID
            # print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
            # plt.imshow(mask, cmap='gray')
            # plt.show()
        # show our output image and connected component mask

    img_out = cv2.bitwise_and(img, img, mask=mask)

    return img_out, mask


def connected_components_not_dicom(img, show_components=False):

    # PNG
    # _, img_thres = cv2.threshold(img, 0, 255, cv2.THRESH_OTSU)
    # _, img_thres = cv2.threshold(img, 0, 255, cv2.THRESH_TOZERO)
    # _, img_thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    _, img_thres = cv2.threshold(img, 22, 255, cv2.THRESH_BINARY)
    # _, img_thres = cv2.threshold(img, 10, 255, cv2.THRESH_BINARY)
    # _, img_thres = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)

    # JPG
    #_, img_thres = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY)

    # show_images(img, img_thres)

    img_thres = img_thres.astype(np.uint8)

    # Apply the Component analysis function
    output = cv2.connectedComponentsWithStats(img_thres,
                                              4,
                                              cv2.CV_32S)
    (totalLabels, labels, stats, centroids) = output

    # Initialize a new image to
    # store all the output components
    mask = np.zeros(img.shape, dtype="uint8")

    # loop over the number of unique connected component labels
    for i in range(1, totalLabels):

        # text = "examining component {}/{}".format(i + 1, totalLabels)
        # print("[INFO] {}".format(text))

        # extract the connected component statistics and centroid for
        # the current label
        # x = stats[i, cv2.CC_STAT_LEFT]
        # y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]

        # print(w)
        # print(h)
        # print(area)

        keep_width = w > 100
        keep_height = h > 100
        keep_area = area > 10000

        if show_components:
            img_copy = img.copy()

            showComponentMask = (labels == i).astype("uint8") * 255

            show_images(img_copy, showComponentMask)

        # ensure the connected component we are examining passes all
        # three tests
        if all((keep_width, keep_height, keep_area)):
            # construct a mask for the current connected component by
            # finding a pixels in the labels array that have the current
            # connected component ID
            # print("[INFO] keeping connected component '{}'".format(i))
            componentMask = (labels == i).astype("uint8") * 255
            mask = cv2.bitwise_or(mask, componentMask)
            # plt.imshow(mask, cmap='gray')
            # plt.show()
        # show our output image and connected component mask
    img_out = cv2.bitwise_and(img, img, mask=mask)

    # img_out = image_segmentation(img_out)

    return img_out, mask
