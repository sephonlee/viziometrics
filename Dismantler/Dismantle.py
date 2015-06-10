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
    
corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat0124_pure/scheme'
# corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/test_corpus/large_corpus'
resultPath = '/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/Result'
resultPath = getModelPath(resultPath, '')
 
Opt_dmtler = Option_Dismantler(isTrain = False)
Dmtler = Dismantler(Opt_dmtler)

nImageAll = 0
startTime = time.time()

countSingle = 0
countMulti = 0

countSplitTime = 0
countMergeTime = 0
countSelectTime = 0

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
                
            subStartTime = time.time()
            
            first_vertical, fire_lane_map_vertical, count_standalone_vertical = Dmtler.split(img, 0)
            first_horizontal, fire_lane_map_horizontal, count_standalone_horizontal = Dmtler.split(img, 1)
    
            spendTime = time.time()
            countSplitTime += spendTime - subStartTime
            subStartTime = spendTime
            print countSplitTime
#             print count_standalone_vertical, count_standalone_horizontal
            first_vertical = Dmtler.merge(img, first_vertical)
            first_horizontal = Dmtler.merge(img, first_horizontal)
            
            spendTime = time.time()
            countMergeTime += time.time() - subStartTime
            subStartTime = spendTime
            print countMergeTime
            
            final_result = Dmtler.select(img, first_vertical, first_horizontal)
            
            spendTime = time.time()
            countSelectTime += time.time() - subStartTime
            subStartTime = spendTime
            print countSelectTime
            
            if len(final_result) <= 1:
                countSingle += 1
            else:
                countMulti += 1
            
            
#             Dmtler.showSegmentationByList(img, node_list)
#             Dmtler.saveSegmentationLayoutByList(img, final_result, resultPath, fname)
            
            nImageAll += 1 
            if nImageAll % 100 == 0:
                print '%d images have been dismantled.' % nImageAll
                            
costTime = time.time() - startTime
print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, resultPath, costTime) 

print float(countSingle)/nImageAll

print "splitTime = %f, avg = %f" %(countSplitTime, float(countSplitTime)/ nImageAll)
print "mergeTime = %f, avg = %f" %(countMergeTime, float(countMergeTime)/ nImageAll)
print "selectTime = %f, avg = %f" %(countSelectTime, float(countSelectTime)/ nImageAll)
