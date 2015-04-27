import sys
sys.path.append("..")
from Dismantler.Dismantler import *

# sys.path.append("/opt/local/Library/Python/2.7/site-packages/")
# import tesseract

import cv2 as cv
import os
import pytesseract
from PIL import Image
import subprocess
from matplotlib import pyplot as plt

# return_code = subprocess.call("tesseract -l eng /Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat1_multi/image_537.jpg  output", shell = False)  


# print subprocess.call("cat output.txt", shell = False)
filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat1_multi/image_1385.jpg"
img = cv.imread(filename)
if len(img.shape) == 3:
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
     

plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
plt.show()
    
batchDim = [60, 60]
shift = 20
imgDim = img.shape
x = 0
y = 0

map = np.zeros(imgDim)
print map

while x <= imgDim[1] - batchDim[1]:
    startX = x
    endX = startX + batchDim[1]
    
    while y <= imgDim[0] - batchDim[0]:
    
        startY = y
        endY = startY + batchDim[0]
        img_batch = img[startY:endY, startX:endX]
        
#         plt.imshow(img_batch, cmap = 'gray', interpolation = 'bicubic')
#         plt.show()
        y = startY + shift
    
        img_batch = Image.fromarray(img_batch)
        text = pytesseract.image_to_string(img_batch) 
        len_text = len(text)
        if len_text > 0:
            print text
            map[startY:endY, startX:endX] += len_text
    
    x = startX + batchDim[1]/2
    y = 0
    


plt.imshow(map, cmap = 'gray', interpolation = 'bicubic')
plt.show()

     
# rows,cols = img.shape
# M = cv.getRotationMatrix2D((cols/2,rows/2), -90,1)
# img = cv.warpAffine(img, M,(cols,rows))


# plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
# plt.show()
# 
# img_ = img[0:20, 0:20]
# img_i = Image.fromarray(img_)
# plt.imshow(img_i, cmap = 'gray', interpolation = 'bicubic')
# plt.show()
# 
# # img = Image.open("/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat1_multi/image_1069.jpg")
# # # Dismantler.showImg(img)
# # 
# # print dir(pytesseract)
# text = pytesseract.image_to_string(img_i) 
# print text
# print len(text)