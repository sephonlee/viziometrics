import cv2 as cv
import os
import numpy as np
from Classifier.Options import *
from Classifier.Models import FeatureDescriptor

corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/TextRecognizer/ee_cat1_multi'
savePath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/TextRecognizer/ee_cat1_patches' 
filename = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/TextRecognizer/ee_cat1_multi/image_377.jpg'
dim = 32


def im2col(Im, block, style='sliding'):

    bx, by = block
    Imx, Imy = Im.shape
    colH = (Imx - bx + 1) * (Imy - bx + 1)
    colW = bx * by 
    imCol = np.zeros((colH, colW))
    curCol = 0
    for j in range(0, Imy):
        for i in range(0, Imx):
            if (i+bx <= Imx) and (j+by <= Imy):
#                 print curCol
                imCol[curCol, :] = Im[i:i+bx, j:j+by].T.reshape(bx*by)
                curCol += 1
            else:
                break
    return np.asmatrix(imCol)


img = cv.imread(filename)
if len(img.shape) == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
patches = im2col(img, (dim, dim))  
print patches.shape




# dim = 32
# for dirPath, dirNames, fileNames in os.walk(corpusPath):   
#     for f in fileNames:
#         fname, suffix = Common.getFileNameAndSuffix(f)
#         if suffix in ['jpg', 'png']:
#             
#             filename = os.path.join(dirPath, f)
#             # Loading Images
#             img = cv.imread(filename)
#             if len(img.shape) == 3:
#                 img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#                 
#             patches = FeatureDescriptor.im2col(img, (dim, dim))
# 
#             for i in patches.shape[2]:
                
