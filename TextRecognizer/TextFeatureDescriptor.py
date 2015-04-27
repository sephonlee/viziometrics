import numpy as np
import cv2 as cv
import pytesseract
import math
import time
from PIL import Image
from matplotlib import pyplot as plt

class TextFeatureDescriptor():
    
    def __init__(self):
        return
    
    @ staticmethod
    def getTextRegion(img, batchDim, shift):
        
        
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        imgDim = img.shape
        map = np.zeros(imgDim)
        
        if imgDim[0] < batchDim[0] or imgDim[1] < batchDim[1]:
            return map
        
        x = 0
        y = 0
        repeat = 0

        while x <= imgDim[1] - batchDim[1]:
            startX = x
            endX = startX + batchDim[1]
            
            while y <= imgDim[0] - batchDim[0]:
            
                startY = y
                endY = startY + batchDim[0]
                img_batch = img[startY:endY, startX:endX]
                
#                 print np.var(img_batch)
#                 plt.imshow(img_batch, cmap = 'gray', interpolation = 'bicubic')
#                 plt.show()
                y = startY + shift
            
                if np.var(img_batch) == 0:
                    len_text = 0
                else:
                    img_batch = Image.fromarray(img_batch)
                    text = pytesseract.image_to_string(img_batch) 
                    
                len_text = len(text)
                if len_text > 0:
#                     print text
                    map[startY:endY, startX:endX] += len_text
            
                repeat += 1
            x = startX + batchDim[1]/2
            y = 0
            
        print "repeat = ", repeat
        return map
    
    @ staticmethod
    def getTextFeatureFromTextRegionMap(map, division):
        
        feature = np.zeros((1, division[0] * division[1]))
        mapDim = map.shape
        if mapDim[0] >= division[0] and mapDim[1] >= division[1]:
            
            step_y = (float(mapDim[0]) / division[0])
            step_x = (float(mapDim[1]) / division[1])
            
            index = 0
            for i in range(0, division[0]):
                
                start_y = round(i * step_y)
                if (i == (division[0] - 1) or (i+1)*step_y >= mapDim[0]):
                    end_y = mapDim[0]
                else:
                    end_y = round((i+1) * step_y)
                
                for j in range(0, division[1]):
                    
                    
                    start_x = round(j * step_x)
                    if (j == (division[1] - 1) or (j+1)*step_x >= mapDim[1]):
                        end_x = mapDim[1]
                    else:
                        end_x = round((j+1) * step_x)
                    
#                     print 'yrange', start_y, end_y
#                     print 'xrange', start_x, end_x
                           
                    subMap = map[start_y:end_y, start_x:end_x]
                    subDim = subMap.shape
#                     plt.imshow(subMap, cmap = 'gray', interpolation = 'bicubic')
#                     plt.show()
#                     print (subMap > 0).sum(), (subDim[0] * subDim[1])
                    
                    
                    textPer = float((subMap > 0).sum()) / (subDim[0] * subDim[1])
#                     print textPer
                    feature[0, index] = textPer
                    index += 1
            
        return feature

    @ staticmethod
    def getImageTextFeatureFromImagePath(fileList, batchDim = (60,60), shift = 60, division = (12,12)):
               
        data = np.zeros([len(fileList), division[0] * division[1]])
    
        for (i,filename) in enumerate(fileList):
            
            img = cv.imread(filename)
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    
            map = TextFeatureDescriptor.getTextRegion(img, batchDim, shift)
            
            
            data[i, :] = TextFeatureDescriptor.getTextFeatureFromTextRegionMap(map, division)
            
            print '%d / %d images (%d, %d)has been collected.' %(i, len(fileList), img.shape[0], img.shape[1])
            
        print '%d images has been collected.' %len(fileList)
        return data
    
    
if __name__ == '__main__':
    
    # print subprocess.call("cat output.txt", shell = False)
    filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat0124_single_composite/composite/image_5752.jpg"
    img = cv.imread(filename)
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
      
    batchDim = [60, 60]
    shift = 40
    division = (12,12)
    startTime = time.time()
    map = TextFeatureDescriptor.getTextRegion(img, batchDim, shift)
       
    plt.imshow(map, cmap = 'gray', interpolation = 'bicubic')
    plt.show()
        
    feature = TextFeatureDescriptor.getTextFeatureFromTextRegionMap(map, division)
    endTime = time.time()
    print "time:", endTime - startTime