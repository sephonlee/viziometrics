# Class to load images
# 1. CloadImageLoader: Load images from s3 server. Usually for testing and classification
# 2. ImageLoader: Load images from local disk. Usually for training 
import sys
sys.path.append("..")
from DataFileTool.DataFileTool import *

import time
import os, errno
import numpy as np
import cv2 as cv
from Options import *

# CloudImageLoader
from boto.s3.connection import S3Connection
from boto.s3.key import Key
import boto
# from PIL import Image
from cStringIO import StringIO


class CloudImageLoader():
    
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
        
#         conn = S3Connection(access_key, secret_key)
        conn = boto.s3.connect_to_region(
           region_name = 'us-west-2',
           aws_access_key_id = access_key,
           aws_secret_access_key = secret_key,
           calling_format = boto.s3.connection.OrdinaryCallingFormat()
           )
        
        self.bucket = conn.get_bucket(host)
        
    def getKey(self, keyname):
        return self.bucket.get_key(keyname)
    
    def getBucketList(self):
        return self.bucket.list()
    
    def isKeyValidImageFormat(self, key):
        keyname =  key.name.split('.')
        if len(keyname) >= 2:
            isValidImageFormat = keyname[-1].lower() in self.Opt.validImageFormat
            isValidSize = key.size < self.Opt.validMinKeySize
            return (isValidImageFormat and isValidSize), keyname[-1]
        else:
            return False, ''
        
    def upLoadingFileToPath(self, keyName, keyPath, filePath):
        print 'Uploading %s as %s' %(filePath, keyName)
        full_key_name = os.path.join(keyPath, keyName)
        k = self.bucket.new_key(full_key_name)
        k.set_contents_from_filename(filePath)
        print 'Complete Uploading'
        
    def upLoadingFile(self, keyName, filePath):
        print 'Uploading %s as %s' %(filePath, keyName)
        k = self.bucket.new_key(keyName)
        k.set_contents_from_filename(filePath)
        print 'Complete Uploading'
        
    @ staticmethod
    def keyToValidImage(key):
        imgStringData = key.get_contents_as_string()
#         fileImgData = StringIO(imgStringData)
#         img = Image.open(fileImgData).convert('RGB')
#         img = np.array(img) 
        
        nparr = np.fromstring(imgStringData, np.uint8)
#         img = cv.imdecode(nparr, cv.CV_LOAD_IMAGE_COLOR)
        img = cv.imdecode(nparr, 1)
        
#         if len(img.shape) == 3:
#             img = img[:, :, ::-1].copy() 
        return img    
    
    def keyToFile(self, keyname, filename):
        key = self.bucket.get_key(keyname)
        keyname =  key.name.split('.')
        suffix = keyname[-1]
        fname = filename + '.' + suffix
        fp = open(fname, "w")
        key.get_file(fp)
        fp.close()
        '% saved'
        return filename
        
    @ staticmethod
    def keyToValidImageOnDisk(key, filename): 

        keyname =  key.name.split('.')
        suffix = keyname[-1]
        fname = filename + '.' + suffix
        fp = open(fname, "w")
        key.get_file(fp)
        fp.close()
        print 'error point'            
        img = cv.imread(fname, 0)
        return img
         
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
        
        for dirName in os.listdir(inPath):
            if dirName in self.Opt.classNames:
                localClassDirs.append(dirName)
                localClassIDs[dirName] = self.Opt.classInfo[dirName]
        
        ####################
        # Loop to read files
        for className in localClassDirs:
            
            classPath = os.path.join(inPath, className)
            fileList = self.getFileNamesFromPath(classPath)
            imData, imDims, dimSum = self.loadImagesByList(fileList, self.Opt.finalDim, self.Opt.preserveAspectRatio)

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
        DataFileTool.saveCSV(outPath, csvFileName, saveContent, header)

        # output
        return allImData, allLabels, allClassNames, localClassDirs
    
    
    @ staticmethod
    def resizeConstantARWithNoEmpty(img, finalDim):
        
        imDim = img.shape
        target_height = finalDim[0]
        target_width = finalDim[1]
        
        origin_height = imDim[0]
        origin_width = imDim[1]
        
        aspect_ratio = float(origin_height) / origin_width;
        
        if aspect_ratio > 1: # tall
            target_width = int(target_height / aspect_ratio)
        else: # wide
            target_height = int(target_width * aspect_ratio)            
        
        return cv.resize(img, (target_width, target_height))

    
    @ staticmethod
    def resizeConstantAR(img, finalDim):
        
        if np.max(img) > 1:
            white_pixel_value = 255;
        else:
            white_pixel_value = 1
            
        imDim = img.shape
        target_height = finalDim[0]
        target_width = finalDim[1]
        
        origin_height = imDim[0]
        origin_width = imDim[1]
        
        aspect_ratio = float(origin_height) / origin_width;
        
        newImg = np.zeros(finalDim) + white_pixel_value
        
        if aspect_ratio > 1: # tall
            tmp_width = int(target_height / aspect_ratio)
            offset_width = (target_width - tmp_width) / 2
            
            tmp_img = cv.resize(img, (tmp_width, target_height))
            newImg[:, offset_width : (offset_width + tmp_width)] = tmp_img
            
        else: # wide
            tmp_height = int(target_width * aspect_ratio)            
            offset_height = (target_height - tmp_height) / 2
            
            tmp_img = cv.resize(img, (target_width, tmp_height))
            newImg[offset_height : (offset_height + tmp_height), :] = tmp_img
        
        return newImg
    
    ##### Duplicate Code in Dismantler #####
    @ staticmethod
    def updateImageToEffectiveArea(img, thresholds = {'splitThres': 0.999, 'varThres': 3, 'var2Thres': 100}):
        heads, ends = ImageLoader.getEffectiveImageArea(img, thresholds)
        
        if ends[1] > heads[1] and ends[0] > heads[0]:
            new_img = img[heads[1]:ends[1], heads[0]:ends[0]]
        else:
            new_img = img
            
        return new_img

    @ staticmethod
    def indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]
    
    @ staticmethod
    def getBlankLine(img, orientation, thresholds): 
        arraySum = np.sum(img, axis = orientation)
        arraySum_nor = arraySum/float(np.max(arraySum))
        arrayVar = np.var(img, axis = orientation) 

        blank_line = ImageLoader.indices(zip(arrayVar, arraySum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
        return blank_line
    
    @ staticmethod 
    def getEffectiveImageArea(img, thresholds):
        
        img_dim = img.shape
        if len(img.shape) == 3:
            img_mono = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_mono = img
        heads = []
        ends = []
        
        for orientation in range(0, 2):
            blank_line = ImageLoader.getBlankLine(img_mono, orientation, thresholds)
            if len(blank_line) > 0:
                if blank_line[0] == 0:
                    head = 0
                    while (head + 1) < len(blank_line):
                        if blank_line[head + 1] - blank_line[head] > 1:
                            break
                        head += 1     
                    heads.append(blank_line[head])
                else:
                    heads.append(0)
                
                if blank_line[-1] == img_dim[(orientation + 1) % 2]-1:
                    end = len(blank_line) - 1
                    while end - 1 >= 0:
                        if blank_line[end] - blank_line[end - 1] > 1:
                            break
                        end -= 1
                    ends.append(blank_line[end])
                else:
                    ends.append(img_dim[(orientation + 1) % 2])
            else:
                heads.append(0)
                ends.append(img_dim[(orientation + 1) % 2])
                
        return heads, ends 
    ##### Duplicate Code in Dismantler #####
    
    
    
    @ staticmethod
    def preImageProcessing(img, finalDim, preserveAR = True):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        img = ImageLoader.updateImageToEffectiveArea(img)
        imDim = img.shape 
        
        if preserveAR:
            img = ImageLoader.resizeConstantAR(img, (finalDim[0], finalDim[1]))
        else:
            img = cv.resize(img, (finalDim[0], finalDim[1]))
            
        img = np.asarray(img)
        img = np.reshape(img, (1, finalDim[0]* finalDim[1]), 'F')
        return img, imDim
        
    # return all images data from the given directory 
    @ staticmethod
    def loadImagesByList(fileList, finalDim, preserveAR):
        
        nx, ny, nz = finalDim;
        imDims = [];

        # read all images and reshaping
        imData = np.zeros((len(fileList), nx*ny*nz), )

        count = 0;
        dimHeightSum = 0
        dimWidthSum = 0
        for filename in fileList:
            img = cv.imread(filename)            
            img, imDim = ImageLoader.preImageProcessing(img, finalDim, preserveAR)
            
            dimHeightSum += imDim[0]
            dimWidthSum += imDim[1]
            imDims.append(imDim[:2])
            dimSum = [dimHeightSum, dimWidthSum]
            
            imData[count, :] = img
            count += 1
            dimSum = [dimHeightSum, dimWidthSum]  

        return imData, imDims, dimSum
    
    # Load images from a list of images, particular use for dismantler
    @ staticmethod
    def loadSubImagesByNodeList(img, nodeList, finalDim, preserveAR):
        
        nx, ny, nz = finalDim;
        imDims = [];

        # read all images and reshaping
        imData = np.zeros((len(nodeList), nx*ny*nz), ) ###

        count = 0;
        dimHeightSum = 0
        dimWidthSum = 0
        for node in nodeList:
            
            start = node.info['start']
            end = node.info['end']
            subImg = img[start[0]:end[0], start[1]:end[1]]
            subImg, imDim = ImageLoader.preImageProcessing(subImg, finalDim, preserveAR)
            
            dimHeightSum += imDim[0]
            dimWidthSum += imDim[1]
            imDims.append(imDim[:2])
            dimSum = [dimHeightSum, dimWidthSum]
            
            imData[count, :] = subImg
            count += 1
            dimSum = [dimHeightSum, dimWidthSum]  

        return imData, imDims, dimSum
        
    @ staticmethod
    def loadImageByPath(filePath):
        img = cv.imread(filePath)         
        return img
        
    
    # Get all valid filenames from the given path
    def getFileNamesFromPath(self, path):

        fileList = []
        num = 0;
        for dirPath, dirNames, fileNames in os.walk(path):   
            for f in fileNames:
                extension = f.split('.')[-1].lower()
                if extension in self.Opt.validImageFormat:
                    fileList.append(os.path.join(dirPath, f))
                    num += 1
                
        return fileList
    
if __name__ == '__main__':   


    Opt = Option(isTrain = True)
    modelPath = Opt.modelPath
#     Opt.saveSetting(modelPath)
    
    ImgLoader = ImageLoader(Opt)
    allImData, allLabels, allCatNames, newClassNames = ImgLoader.loadTrainDataFromLocalClassDir(Opt.trainCorpusPath, modelPath)      
