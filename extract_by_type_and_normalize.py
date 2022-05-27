#!/usr/bin/env python3

import shutil
import cv2
import numpy as np

import pandas as pd

# -----------------------------------------------------------------
# HAM10000
# -----------------------------------------------------------------
SOURCE_FOLDER = '/Volumes/Project/Lapisco/datasets/HAM10000/Data'
DESTINATION_FOLDER = '/Volumes/Project/Lapisco/datasets/HAM10000/DataNormalized'
DESTINATION_FOLDER_NORMALIZED = '/Volumes/Project/Lapisco/datasets/HAM10000/TrainingNormalized'
df = pd.read_csv('HAM10000_metadata.csv')


def normalize_image(input_path, output_directory):
    img = cv2.imread(input_path)
    imgBase = cv2.imread('/Volumes/Project/Lapisco/datasets/HAM10000/HAIR_Test/Base.jpg')

    _imgBase = np.copy(imgBase)
    _img = np.copy(img)

    meanBase = np.mean(_imgBase)
    meanImg = np.mean(_img)
    _img = _img * (meanBase / meanImg)

    cv2.imwrite(output_directory, _img)


def remove_hair(input_image, output_directory):
    src = cv2.imread(input_image)

   
    grayScale = cv2.cvtColor(src, cv2.COLOR_RGB2GRAY)

 
    kernel = cv2.getStructuringElement(1, (17, 17))

    blackhat = cv2.morphologyEx(grayScale, cv2.MORPH_BLACKHAT, kernel)

    ret, thresh2 = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

    dst = cv2.inpaint(src, thresh2, 1, cv2.INPAINT_TELEA)
    cv2.imwrite(output_directory, dst, [int(cv2.IMWRITE_JPEG_QUALITY), 90])


for index, row in df.iterrows():
    print("{}/{}.jpg".format(SOURCE_FOLDER, row['image_id']))

    remove_hair("{}/{}.jpg".format(SOURCE_FOLDER, row['image_id']), "{}/{}.jpg".format(DESTINATION_FOLDER, row['image_id']))
    normalize_image("{}/{}.jpg".format(DESTINATION_FOLDER, row['image_id']), "{}/{}.jpg".format(DESTINATION_FOLDER, row['image_id']))

    shutil.copy("{}/{}.jpg".format(DESTINATION_FOLDER, row['image_id']),
                "{}/{}/{}.jpg".format(DESTINATION_FOLDER_NORMALIZED, row['dx'], row['image_id']))
