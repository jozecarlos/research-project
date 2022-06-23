import sys, os
import numpy as np
import cv2
import skimage
from scipy.ndimage import gaussian_gradient_magnitude
from scipy.ndimage import  generic_gradient_magnitude
from skimage.color import label2rgb
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
from scipy.io import loadmat
import shutil
import csv
from skimage.measure import *
from skimage.morphology import disk
from skimage.filters.rank import entropy
from scipy.stats import multivariate_normal
from scipy import ndimage as ndi
import time
from skimage import feature
rt = "C:/Users/Solon/Pictures/datasets/ISIC2018_VisionTraining"
tester = "C:/Users/Solon/Pictures/datasets/ISIC2018_VisionTraining_Data"



fileLBP = open('FE_LBP.dat', 'w')


def vision3(image , vaal):
    imageInput = (np.double(image) + 1) * 2
    try:
        imageInput = np.double(imageInput[:, :, 2])
    except:
        imageInput = imageInput

    lbp_image = local_binary_pattern(imageInput, 4, 1, "uniform")
    image = imageInput + np.double(lbp_image)


    return  np.uint8(np.log(image)  / np.log(vaal))

def normalizeImage(v):
  v = (v - v.min()) / (v.max() - v.min())
  result = (v * 255).astype(np.uint8)
  return result




img1 = cv2.imread("/media/sol2/Novo volume/Users/Solon/PycharmProjects/ISIC_2018/ISIC_0000000.jpg",0)

import cv2

im = vision3(img1,2.30)

cv2.imshow('  Normal2 ', normalizeImage(im) )

k = cv2.waitKey(0)
