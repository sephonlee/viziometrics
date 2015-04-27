import sys
sys.path.append("..")

from  Classifier.Options import *
from  Options_Dismantler import *
from Dismantler import *

import cv2 as cv
import os
import time

def getModelPath(path, dirName):
    dirName = dirName + datetime.datetime.now().strftime("%Y-%m-%d")
    return Common.makeDir(path, dirName)
    
    
# corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/test_corpus/large_corpus'
# corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat0124_pure/equation'
corpusPath = '/Users/sephon/Desktop/Research/ReVision/corpus/pm_ee_classified/cat_0/mix'
resultPath = '/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/Result'
resultPath = getModelPath(resultPath, '')
 
Opt_dmtler = Option_Dismantler(isTrain = False)
Dmtler = Dismantler(Opt_dmtler)

nImageAll = 0
startTime = time.time()

for dirPath, dirNames, fileNames in os.walk(corpusPath):   
    for f in fileNames:
        fname, suffix = Common.getFileNameAndSuffix(f)
        if suffix in Opt_dmtler.validImageFormat:
            
            filename = os.path.join(dirPath, f)
            # Loading Images
            img = cv.imread(filename)
            print filename
#             node_list = Dmtler.dismantle(img)
            
            if len(img.shape) == 3:
                img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        
            root0, count_standalone = Dmtler.split(img, 0)
            map0 = Dmtler.getSalientMap(img, root0)
            
            root1, count_standalone = Dmtler.split(img, 1)
            map1 = Dmtler.getSalientMap(img, root1)
            
            map = (map0 + map1) / 2
            
#             Dismantler.showImg(map)
            savePath = os.path.join(resultPath, f)
#             print savePath
            cv.imwrite(savePath, map)
            
            nImageAll += 1 
            if nImageAll % 100 == 0:
                print '%d images have been mapped.' % nImageAll
                            
costTime = time.time() - startTime
print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, resultPath, costTime) 

