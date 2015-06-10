import sys
sys.path.append("..")

from Classifier.DataManager import *
from Classifier.Models import SVMClassifier
from Dismantler import *
from Options_Dismantler import *
from scipy.cluster.vq import whiten
# from TextRecognizer.TextFeatureDescriptor import *

import cv2 as cv
import numpy as np


def normalizeAndWhiten(A):
                
    pM = np.asmatrix(np.mean(A, axis=1)).T
    pSqVar = np.asmatrix(np.sqrt(A.var(axis=1) + 10)).T
    pSqVar = pSqVar.astype('float64') 
    A = np.divide((A - pM),  pSqVar)

    # whiten
    C = np.cov(A, rowvar = False)
    M = np.mean(A, axis = 0)
    D,V = np.linalg.eig(C)
    P = np.dot(np.dot(V, np.diag(np.sqrt(1/(D + 0.1)))), V.T)
    A = np.dot(A - M, P)

    return A, M, P

# def whiten(X):
#     X = np.dot((X - self.M), self.P)

def svd_whiten(X):

    U, s, Vt = np.linalg.svd(X)

    # U and Vt are the singular matrices, and s contains the singular values.
    # Since the rows of both U and Vt are orthonormal vectors, then U * Vt
    # will be white
    print X.shape
    print U.shape
    print Vt.shape
    X_white = np.dot(U, Vt)

    return X_white




def getLabeledName(labeled_names, quantities):
    
    allLabeledNames = []
    for label_name, quantity in zip(labeled_names, quantities):
        allLabeledNames = np.hstack([allLabeledNames, np.tile(label_name, quantity)])
        print '%d %s has been generated in the array' %(quantity, label_name)
    return allLabeledNames


def getImageTextFeatureFromImagePath(fileList, division):
           
    
    data = np.zeros([len(fileList), division[0] * division[1]])

    for (i,filename) in enumerate(fileList):
        
        img = cv.imread(filename)
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img = cv.resize(img, division)
        data[i, :] = np.reshape(img, (1, division[0]* division[1]), 'F')
        
    print '%d images has been collected.' %len(fileList)
    return data

def getFeatureByFireLaneFromFileList(CID, fileList):
    
    data = np.zeros([len(fileList), 2 + CID.num_cut*2])
    for (i,filename) in enumerate(fileList):
        
        img = cv.imread(filename)
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        data[i, :] = CID.getFeatureByFireLane(img)
        
        print '%d / %d images has been collected' %(i, len(fileList))
        
    print '%d images has been collected.' %len(fileList)
    return data
    


def getFeatureByFireLaneMapFromFileList(CID, fileList, showStatus = True):
    
    Opt_Dmtler = Option_Dismantler(isTrain = False)
    Dmtler = Dismantler(Opt_Dmtler)
    
    data = np.zeros([len(fileList), 2 + CID.division[1] * CID.division[0]])
    for (i,filename) in enumerate(fileList):
        
        img = cv.imread(filename)
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        first_vertical, fire_lane_map_vertical, count_standalone_first_vertical = Dmtler.split(img, 0)
        first_horizontal, fire_lane_map_horizontal, count_standalone_first_horizontal = Dmtler.split(img, 1)
        map = fire_lane_map_vertical + fire_lane_map_horizontal
#         Dmtler.showImg(fire_lane_map_vertical)
#         Dmtler.showImg(fire_lane_map_horizontal)
#         Dmtler.showImg(map)
        
        data[i, :] = CID.getFeatureByFireLaneMap(map)
        
        if showStatus:
            print '%d / %d images has been collected' %(i, len(fileList))
        
    print '%d images has been collected.' %len(fileList)
    return data

def getTrainImageAverageDim(filenames_array):
    
    count = 0
    height = 0
    width = 0
    for filenames in filenames_array:
        for filename in filenames:
            img = cv.imread(filename)
            imgDim = img.shape
            height += imgDim[0]
            width += imgDim[1]
            count += 1
            
    return (float(height)/count, float(width)/count)
        
    

if __name__ == '__main__':
    
    
    Opt_CID = Option_CompositeDetector(isTrain = True)
    ImgLoader = ImageLoader(Opt_CID)
    CID = CompositeImageDetector(Opt_CID)
    
#     print finalDim
#     singleImageFileList = "/home/ec2-user/VizioMetrics/Corpus/Dismantler/train_single_composite/single"
#     compositeImageFileList = "/home/ec2-user/VizioMetrics/Corpus/Dismantler/train_single_composite/composite"
    
    singleImageFileList = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat0124_single_composite/single"
    compositeImageFileList = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat0124_single_composite/composite"
    
    print 'Loading images...'
    compositeImageFileList = ImgLoader.getFileNamesFromPath(compositeImageFileList)
    singleImageFileList = ImgLoader.getFileNamesFromPath(singleImageFileList)
    
    
#     print getTrainImageAverageDim([compositeImageFileList, singleImageFileList])
    
    print '%d single images and %d composite images' %(len(compositeImageFileList), len(singleImageFileList))
    print 'Featuring...'
#     allFeatures = np.vstack([getFeatureByFireLaneFromFileList(CID, compositeImageFileList), getFeatureByFireLaneFromFileList(CID, singleImageFileList)])
    allFeatures = np.vstack([getFeatureByFireLaneMapFromFileList(CID, compositeImageFileList), getFeatureByFireLaneMapFromFileList(CID, singleImageFileList)])
    
    print allFeatures.shape
#     allFeatures, M, P = normalizeAndWhiten(allFeatures);
    CID.saveFeatureDescriptorParamToFile(Opt_dmtler.modelPath)
    
#     allFeatures = whiten(allFeatures)
    allLabeledNames = getLabeledName(Opt_dmtler.classNames, [len(compositeImageFileList), len(singleImageFileList)])
    print 'All images are loaded'
    
    
    print allFeatures[:,0:2].shape
    SVM_train = SVMClassifier(Opt_dmtler, isTrain = True)
    SVM_train.trainModel(allFeatures[:,0:2], allLabeledNames)
#     
#     
#     SVM_train = SVMClassifier(Opt_train, isTrain = True)
#     SVM_train.trainModel(allFeatures, allLabeledNames)
    print 'Model has been trained'
    
    


