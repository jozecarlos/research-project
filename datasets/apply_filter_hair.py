#!/usr/bin/env python3

import cv2
import numpy as np

# SOURCE_FOLDER = '/Volumes/Project/Lapisco/datasets/HAM10000/HAIR_Test/ISIC_0024384.jpg'
SOURCE_FOLDER = '/Volumes/Project/Lapisco/datasets/HAM10000/HAIR_Test/ISIC_0000043.jpg'
DESTINATION_FOLDER = '/Volumes/Project/Lapisco/datasets/HAM10000/HAIR_Test/'


def resizer(input_path, output_directory):
    img = cv2.imread(input_path)
    new_width = int(img.shape[1]/2)
    new_height = int(img.shape[0]/2)

    img_half = cv2.resize(img, (new_width, new_height))
    cv2.imwrite("{}/InPainted_sample1_resized.jpg".format(output_directory), img_half)


def normalize(input_path, output_directory):
    img = cv2.imread(input_path)
    norm_img = np.zeros((800, 800))
    final_img = cv2.normalize(img, norm_img, 0, 255, cv2.NORM_MINMAX)
    # cv2.imshow('Normalized Image', final_img)
    cv2.imwrite("{}/InPainted_sample1_normalized.jpg".format(output_directory), final_img)


def normalize_image(input_path, output_directory):
    img = cv2.imread(input_path)
    imgBase = cv2.imread('/Volumes/Project/Lapisco/datasets/HAM10000/HAIR_Test/Base.jpg')

    _imgBase = np.copy(imgBase)
    _img = np.copy(img)

    meanBase = np.mean(_imgBase)
    meanImg = np.mean(_img)
    _img = _img * (meanBase / meanImg)

    cv2.imwrite("{}/InPainted_sample1_normalized.jpg".format(output_directory), _img)


def remove_hair(input_image, output_directory):
    src = cv2.imread(input_image)

    # print(src.shape)
    # cv2.imshow("original Image", src)

    # Convert the original image to grayscale
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)
    cv2.imshow("GrayScale", grayScale)
    cv2.imwrite('grayScale_sample1.jpg', grayScale, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # Kernel for the morphological filtering
    kernel = cv2.getStructuringElement(1, (17, 17))

    # Perform the blackHat filtering on the grayscale image to find the
    # hair countours
    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)
    cv2.imshow("BlackHat", blackhat)
    cv2.imwrite('blackhat_sample1.jpg', blackhat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # intensify the hair countours in preparation for the inpainting
    # algorithm
    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
    # print(thresh2.shape)
    # cv2.imshow("Thresholded Mask", thresh2)
    cv2.imwrite('thresholded_sample1.jpg', thresh2, [int(cv2.IMWRITE_JPEG_QUALITY), 90])

    # inpaint the original image depending on the mask
    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
    # cv2.imshow("InPaint", dst)
    cv2.imwrite("{}/InPainted_sample1.jpg".format(output_directory), dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


# remove_hair(SOURCE_FOLDER, DESTINATION_FOLDER)
# normalize_image("{}/InPainted_sample1.jpg".format(DESTINATION_FOLDER), DESTINATION_FOLDER)
resizer(SOURCE_FOLDER, DESTINATION_FOLDER)
