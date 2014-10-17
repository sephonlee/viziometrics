import time
import os, errno
import numpy as np
import cv2 as cv
import csv

import boto
from boto.s3.connection import S3Connection
from boto.s3.key import Key

from Options import *
import multiprocessing as mp


class cloudImageLoader():
    
    Opt = None
    bucket = None
    
    def __init__(self, Opt, keyPath = None, host = None):
        self.Opt = Opt
        if keyPath is None:
            keyPath = self.Opt.keyPath
            
        if host is None:
            host = self.Opt.host
        
        f = open(keyPath, 'r')
        access_key = f.readline()[0:-1]
        secret_key = f.readline()
        
        conn = S3Connection(access_key, secret_key)
        self.bucket = conn.get_bucket(host)
        
    def getBucketList(self):
        return self.bucket.list()
    
    def isKeyValidImageFormat(self, key):
        keyname =  key.name.split('.')
        if len(keyname) == 2:
            return keyname[1] in self.Opt.validImageFormat, keyname[1]
        else:
            return False, ''
        
    def keyToValidImage(self, key):
        
#         filename, suffix= Common.getFileNameAndSuffix(key.name)
#         if key.name.split('.')[1] in self.Opt.validImageFormat:
            
        
        
#         print filename
#         fname = "img.jpg"
#         fp = open (fname, "w")
# #         key.get_file(fp)
#         fp.close
#         
#         image = cv.imread(fname)
        return 

# Class of Image Data
class ImageLoader():
    
    Opt = None
     
    def __init__(self, Opt):
        self.Opt = Opt
        
    # Load training image data from local database
    def loadTrainDataByQuerry(self):
        return
    
    # Load testing image data from local database
    def loadTestDataByQuerry(self):
        self.loadTrainDataByQuerry()
        return
    
    
    # Load testing image data from local classified directories
    # Directory name = class name
    def loadTestDataFromLocalClassDir(self, inPath = None, outPath = None):
        if inPath is None:
            inPath = self.Opt.testCorpusPath##
        if outPath is None:
            outPath = self.Opt.svmModelPath ################################################## need modify
        return self.loadDataFromLocalClassDir(inPath, outPath, mode = 'test')
        
    # Load training image data from local classified directories
    # Directory name = class name
    def loadTrainDataFromLocalClassDir(self, inPath = None, outPath = None):
        if inPath is None:
            inPath = self.Opt.trainCorpusPath
        if outPath is None:
            outPath = self.Opt.modelPath
        return self.loadDataFromLocalClassDir(inPath, outPath, mode = 'train')
        
    def loadDataFromLocalClassDir(self, inPath, outPath, mode = 'unknown'):
        print "Loading Images from" + inPath
        startTime =  time.time()
        
        # Local classes information
        allImData = None
        localClassDirs = [] # Local labeled classe_names
        localClassIDs = {} # Local labeled class_ids
        
        # Local images in classes information
        imDimsInClass = {}
        numImageInClass = {}
        
        fileNamesInClass = []
        allLabels = []
        allClassNames = []
        
        ## Image data overview
        dimHeightSum = 0
        dimWidthSum = 0
        NtotalImages = 0
        maxCatSize = 0
        
        # get categories from directory

        for dirName in  os.listdir(inPath):
            if dirName in self.Opt.classNames:
                localClassDirs.append(dirName)
                localClassIDs[dirName] = self.Opt.classInfo[dirName]
        
        
        ####################
        # Loop to read files
        for className in localClassDirs:
            
            classPath = os.path.join(inPath, className)
            fileList = self.getFileNamesFromPath(classPath)
            imData, imDims, dimSum = self.loadImages(fileList, self.Opt.finalDim)

            NImages = len(imDims)
            
            # stack all images #
            if allImData is None:
                allImData =  imData
            else:
                allImData = np.vstack([allImData, imData])
            #####################
            
            ##
            imDimsInClass[className] = imDims
            numImageInClass[className] = NImages
            ##
            fileNamesInClass += fileList
            allLabels = np.hstack([allLabels, np.tile(localClassIDs[className], (numImageInClass[className]))])
            allClassNames = np.hstack([allClassNames, np.tile(className, (numImageInClass[className]))])
                
            ## Image data overview
            NtotalImages += NImages 
            if NImages > maxCatSize:
                maxCatSize = NImages
            dimHeightSum += dimSum[0]
            dimWidthSum += dimSum[1]
            meanDim = [dimHeightSum / float(NtotalImages), dimWidthSum / float(NtotalImages)]

            print 'Collect', NImages, 'images from', className

        endTime = time.time()
        
        print 'Total', NtotalImages,'images collected in', endTime-startTime, 'sec'
        print 'Average image dimension: ', meanDim, '\n'
         
        
        saveContent = zip(fileNamesInClass, allClassNames)
        csvFileName = mode + '_image_files'
        header = ['file_path', 'class_name']
        Common.saveCSV(outPath, csvFileName, saveContent, header)

        # output
        return allImData, allLabels, allClassNames
    
    
    @ staticmethod
    def preImageProcessing(img, finalDim):
        
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        imDim = img.shape 

        img = cv.resize(img, (finalDim[0], finalDim[1]))
        img = np.asarray(img)
        img = np.reshape(img, (1, finalDim[0]* finalDim[1]))
        
        return img, imDim
    
    
    # return all file paths from the given directory with given file Type
    @ staticmethod
    def loadImages(fileList, finalDim):
        
        nx, ny, nz = finalDim;
        imDims = [];

        # read all images and reshaping
        imData = np.zeros((len(fileList), nx*ny*nz), )

        count = 0;
        dimHeightSum = 0
        dimWidthSum = 0
        for filename in fileList:
            
            img = cv.imread(filename)
            img, imDim = ImageLoader.preImageProcessing(img, finalDim)
            
            dimHeightSum += imDim[0]
            dimWidthSum += imDim[1]
            imDims.append(imDim[:2])
            dimSum = [dimHeightSum, dimWidthSum]
            
            imData[count, :] = img
            count += 1
            dimSum = [dimHeightSum, dimWidthSum]  

        return imData, imDims, dimSum
         
    # Get all valid filenames from the given path
    def getFileNamesFromPath(self, path):

        fileList = []
        num = 0;
        for dirPath, dirNames, fileNames in os.walk(path):   
            for f in fileNames:
                extension = f.split('.')[1]
                if extension in self.Opt.validImageFormat:
                    fileList.append(os.path.join(dirPath, f))
                
        return fileList
        
if __name__ == '__main__':   


    Opt = Opt(isTrain = True)
    modelPath = Opt.modelPath
    Opt.saveSetting(modelPath)
    
    ImgLoader = ImageLoader(Opt)
    allImData, allLabels, allCatNames = ImgLoader.loadTrainDataFromLocalClassDir(Opt.trainCorpusPath, modelPath)