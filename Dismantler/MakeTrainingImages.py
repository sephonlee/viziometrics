import sys
sys.path.append("..")

from  Classifier.Options import *
from  Options_Dismantler import *
from Dismantler import *

import shutil
import cv2 as cv
import os
import time


corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat1_multi'
trainImagePath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/py_split'

Opt_clf = Option_Dismantler(isTrain = False)
# Dmtler = Dismantler(Opt_clf)
# 
# nImageAll = 0
# startTime = time.time()
# 
# for dirPath, dirNames, fileNames in os.walk(corpusPath):   
#     for f in fileNames:
#         fname, suffix = Common.getFileNameAndSuffix(f)
#         if suffix in Opt_clf.validImageFormat:
#             
#             filename = os.path.join(dirPath, f)
#             # Loading Images
#             img = cv.imread(filename)
#             if len(img.shape) == 3:
#                 img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#             
#             node_list = Dmtler.makeTrainData(img, trainImagePath, filename.split('/')[-1], pre_classified = True)
# 
#             
#             nImageAll += 1 
#             if nImageAll % 100 == 0:
#                 print '%d images have been dismantled.' % nImageAll
#                             
# costTime = time.time() - startTime
# print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, trainImagePath, costTime) 

startTime = time.time()
residualImagePath = os.path.join(trainImagePath, 'auxiliary_residual')
trainImagePath = os.path.join(trainImagePath, 'auxiliary')
print residualImagePath
nImageAll = 0
for dirPath, dirNames, fileNames in os.walk(trainImagePath):   
    for f in fileNames:
        fname, suffix = Common.getFileNameAndSuffix(f)
        if suffix in Opt_clf.validImageFormat:
            fname_frag = fname.split('_')
            level = len(fname_frag) - 3
             
            print f
            print os.path.join(residualImagePath, f)
    #         
            if level == 3:
                filename = os.path.join(dirPath, f)
#                 newName = os.path.join(residualImagePath, f)
#                 shutil.copy2(filename, residualImagePath)
                  
                nImageAll += 1 
            if nImageAll % 100 == 0:
                print '%d images have been copied.' % nImageAll
                             
costTime = time.time() - startTime
print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, trainImagePath, costTime) 

