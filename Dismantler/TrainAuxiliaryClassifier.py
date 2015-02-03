
import cv2 as cv
import numpy as np
import math
from matplotlib import pyplot as plt

from Classifier.DataManager import *
from Options_Dismantler import *
from Classifier.Models import SVMClassifier
from Dismantler import SubImageClassifier

def getImageID(filename):
    
    filename = filename.split('/')[-1]
    prefix = filename.split('.')[0]
    return prefix.split('_')[1]

def getOriginalImageInfo(originalImageFileList):
    originalImageInfo = {}
    for filename in originalImageFileList:
        img = cv.imread(filename)
        originalImageInfo[getImageID(filename)] = SubImageClassifier.getImageInfo(img)
        
    return originalImageInfo

def getImageFeatureFromImagePath(imagePath, originalImageInfo, thresholds, num_cut = 5):
    
    data = np.zeros([len(imagePath), 5+num_cut*2])

    for (i,filename) in enumerate(imagePath):
        
        id = getImageID(filename)
        img = cv.imread(filename)
        
        thisOriginalImageInfo = originalImageInfo[id]
        data[i, :] = SubImageClassifier.getImageFeature(thisOriginalImageInfo, img, thresholds, num_cut)

    print '%d images has been collected.' %len(imagePath)
    return data


def getLabeledName(labeled_names, quantities):
    
    allLabeledNames = []
    for label_name, quantity in zip(labeled_names, quantities):
        allLabeledNames = np.hstack([allLabeledNames, np.tile(label_name, quantity)])
        print '%d %s has been generated in the array' %(quantity, label_name)
    return allLabeledNames



if __name__ == '__main__':
    
    Opt_train = Option_Dismantler(isTrain = True)
    ImgLoader = ImageLoader(Opt_train)
    
    originalImagePath = "/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/train_corpus/ee_cat1_multi"
    standaloneImagePath = "/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/train_corpus/ee_cat1_multi_subimages/independent"
    auxiliaryImagePath = "/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/train_corpus/ee_cat1_multi_subimages/merge"
    
    originalImageFileList = ImgLoader.getFileNamesFromPath(originalImagePath)
    standaloneImageFileList = ImgLoader.getFileNamesFromPath(standaloneImagePath)
    auxiliaryImageFileList = ImgLoader.getFileNamesFromPath(auxiliaryImagePath)
    
    originalImageInfo = getOriginalImageInfo(originalImageFileList)
    
    allFeatures = np.vstack([getImageFeatureFromImagePath(standaloneImageFileList, originalImageInfo, Opt_train.thresholds), getImageFeatureFromImagePath(auxiliaryImageFileList, originalImageInfo, Opt_train.thresholds)])
    allLabeledNames = getLabeledName(Opt_train.classNames, [len(standaloneImageFileList), len(auxiliaryImageFileList)])
    
    SVM_train = SubImageClassifier(Opt_train, isTrain = True)
    SVM_train.trainModel(allFeatures, allLabeledNames)
#     
#     
#     SVM_train = SVMClassifier(Opt_train, isTrain = True)
#     SVM_train.trainModel(allFeatures, allLabeledNames)
    print 'Model has been trained'
    
    
    