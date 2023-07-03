from processing import *
import cv2
import os
import numpy as np
from tqdm import tqdm

if __name__ == '__main__':

    images_dicom = []
    images_other = []

    image_path = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/1.dicom"
    image_path2 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/2.dicom"
    image_path3 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/3.dcm"
    image_path4a = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/4a.dicom"
    image_path4b = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/4b.dicom"
    image_path5a = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/5a.dcm"
    image_path5b = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/5b.dcm"
    image_path6 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/dicom/6.dcm"
    image_path10 = "/home/dimatkchnk/Downloads/e92f99ecdf4166ed6203897443df6b2b.dicom"

    images_dicom.append(image_path)
    images_dicom.append(image_path2)
    images_dicom.append(image_path3)
    images_dicom.append(image_path4a)
    images_dicom.append(image_path4b)
    images_dicom.append(image_path5a)
    images_dicom.append(image_path5b)
    images_dicom.append(image_path6)
    images_dicom.append(image_path10)

    image_path_png = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/train/1/32_DDSM.png"
    image_path_png2 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/Dataset of Mammography with Benign Malignant Breast Masses/INbreast Dataset/Benign Masses/20586908 (3).png"
    image_path_png3 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/Dataset of Mammography with Benign Malignant Breast Masses/DDSM Dataset/Benign Masses/D1_A_1177_1.RIGHT_CC (3).png"
    image_path_png4 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/CSAW-S/CsawS/anonymized_dataset/002/002_0.png"
    image_path_png5 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/test/1/959_DDSM.png"
    image_path_png6 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/train/1/40_DDSM.png"
    image_path_png7 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/train/3/15_DDSM.png"
    image_path_png8 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/train/3/64_DDSM.png"
    image_path_png9 = "/home/dimatkchnk/praca_dyplomowa/images/train-for-unet/images/66e6c6f2c85d87f274bbbed22532803e.png"

    image_path_jpg = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/archive/BIRAD1/b1/2013_BC000541_ MLO_L.jpg"
    image_path_jpg2 = "/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/archive/BIRAD1/b1/2019_BC0021581_ MLO_R.jpg"

    images_other.append(image_path_png)
    images_other.append(image_path_png2)
    images_other.append(image_path_png3)
    images_other.append(image_path_png4)
    images_other.append(image_path_png5)
    images_other.append(image_path_png6)
    images_other.append(image_path_png7)
    images_other.append(image_path_png8)
    images_other.append(image_path_png9)
    images_other.append(image_path_jpg)
    images_other.append(image_path_jpg2)

    for image_pth in images_dicom:
        img = load_image_pydicom(image_pth, voi_lut=True)
        img_out, mask = connected_components(img, show_components=False)
        # show_hist(img)
        show_images(img, img_out)
        # convert_to_16_bit_png_file(mask, "test.png")

    # for image_pth in images_other:
    #     img2 = cv2.imread(image_pth, 0)
    #     img_out2, mask = connected_components_not_dicom(img2, False)
    #     # img_out2 = image_segmentation(img_out2)
    #     # show_hist(img2)
    #     show_images(img2, img_out2)
    #     # show_images(img2, mask)

    # img2 = cv2.imread(image_path_jpg2, 0)
    # img_out2 = connected_components_not_dicom(img2, False)
    # show_images(img2, img_out2)

    # FOR PNG/JPG/JPEG
    # Path to the current folder
    # folder_path = f'/home/dimatkchnk/praca_dyplomowa/images/samples-from-databases/ddsm_four_classes/train/1'
    #
    # output_folder = f'/home/dimatkchnk/praca_dyplomowa/images/train-for-unet'
    #
    # # Iterate through each file in the folder
    # for filename in os.listdir(folder_path)[:100]:
    #     file_path = os.path.join(folder_path, filename)  # Full path to the current file
    #
    #     print("Processing: " + file_path.split("/")[-1])
    #
    #     img = cv2.imread(file_path, 0)
    #     img_out, mask = connected_components_not_dicom(img, show_components=False)
    #     # img_out2 = image_segmentation(img_out2)
    #
    #     cv2.imwrite(os.path.join(output_folder, filename), img)
    #     cv2.imwrite(os.path.join(output_folder, filename.split(".")[0] + "_mask" + ".png"), mask)

    # FOR DICOM
    # Path to the current folder
    # folder_path = f'/home/dimatkchnk/praca_dyplomowa/images/dicom-for-unet-bcd'
    # folder_path = f'/home/dimatkchnk/praca_dyplomowa/images/dicom-for-unet-vindr/physionet.org/files/vindr-mammo/1.0.0/images'
    #
    output_folder = f'/home/dimatkchnk/praca_dyplomowa/images/train-for-unet'
    #
    # # Iterate through each file in the folder
    # for folder in tqdm(os.listdir(folder_path), colour="blue"):
    #     child_folder_path = os.path.join(folder_path, folder)
    #
    #     for filename in os.listdir(child_folder_path):
    #         file_path = os.path.join(child_folder_path, filename)  # Full path to the current file
    #
    #         # print("Processing: " + file_path.split("/")[-1])
    #
    #         img = load_image_pydicom(file_path, voi_lut=True)
    #         img_out, mask = connected_components(img, show_components=False)
    #         # img_out2 = image_segmentation(img_out2)
    #
    #         convert_to_16_bit_png_file(img, os.path.join(output_folder, filename.split(".")[0] + ".png"))
    #         convert_to_16_bit_png_file(mask, os.path.join(output_folder, filename.split(".")[0] + "_mask" + ".png"))

    # for filename in tqdm(os.listdir(output_folder), colour="blue"):
    #     file_path = os.path.join(output_folder, filename)  # Full path to the current file
    #
    #     if filename.endswith(".png"):
    #
    #         if 'mask' in filename:
    #             img = cv2.imread(file_path, 0)
    #             convert_to_16_bit_png_file(img, os.path.join(output_folder, "masks", filename))
    #         else:
    #             img = cv2.imread(file_path, 0)
    #             convert_to_16_bit_png_file(img, os.path.join(output_folder, "images", filename))



