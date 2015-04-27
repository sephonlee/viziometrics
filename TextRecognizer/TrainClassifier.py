import sys
sys.path.append("..")

import cPickle
import gzip
import os
import time

import numpy

# import theano
# import theano.tensor as T
import matplotlib.pyplot as plt
import cv2 as cv


from Options_TextRecognizer import *
from Classifier.DataManager import *
from Classifier.Dictionary import *
from Classifier.Models import FeatureDescriptor
from OneClassSVMClassifier import *

def loadChars74KDataset(corpus_path):
    
    fileList = []
    num = 0;
    for dirPath, dirNames, fileNames in os.walk(corpus_path):   
        for f in fileNames:
            extension = f.split('.')[1]
#             if extension in self.Opt.validImageFormat:
#                 fileList.append(os.path.join(dirPath, f))
                    
    return 


## Train Model
Opt_train = Option_TextRecognizer(isTrain = True)
Opt_train.saveSetting()
ImageLoader_train = ImageLoader(Opt_train)

allImData, allLabels, allCatNames, newClassNames = ImageLoader_train.loadTrainDataFromLocalClassDir(Opt_train.trainCorpusPath)
Opt_train.updateClassNames(newClassNames)

Dictionary_train = DictionaryExtractor(Opt_train)
dicPath = Dictionary_train.getLocalDictionaryPath(allImData, allLabels)
# Dictionary_train.showCentroids()

FD_train= FeatureDescriptor(dicPath)
X = FD_train.extractFeatures(allImData, 0)
y = np.ones((allCatNames.shape[0], 1))


SVM_train = OneClassSVMClassifier(Opt_train, isTrain = True)
SVM_train.trainModel(X, y)

print X.shape


# img = cv.imread('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/textOCR/English/Fnt/Sample018/img018-00012.png')
# img = cv.resize(img, (8,8))
# plt.imshow(img, interpolation = 'bicubic')
# plt.show()