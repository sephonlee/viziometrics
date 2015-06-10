import csv
import multiprocessing as mp
import os, errno
import time
from boto.s3.key import Key

from Dictionary import *
from Options import *
from DataManager import *
from Models import *


import sys
sys.path.append("..")
from Dismantler.Dismantler import *
from Dismantler.Options_Dismantler import *

## Global Object   
Opt = Option(isClassify = True)
FD = FeatureDescriptor(Opt.dicPath)
Clf = SVMClassifier(Opt, clfPath = Opt.svmModelPath)
cIL = CloudImageLoader(Opt)
Opt_dmtler = Option_Dismantler(isTrain = False)
Opt_CD = Option_CompositeDetector(isTrain = False)
Dmtler = Dismantler(Opt_dmtler)
CDetector = CompositeImageDetector(Opt_CD)


def listener(name, q, outPath, outFilename):
    print '%s Listener set up in %s' % (name, mp.current_process().name)
    startTime = time.time()
    outFilePath = os.path.join(outPath, outFilename) + '.csv'
    count = 0
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    while True:
        content = q.get()
        if content is not None:
            count += 1
        # Stop
        if content == 'kill':
            costTime = time.time() - startTime
            print'%d images have been classified in %d sec. Stop Listener in %s\n' % (count - 1, costTime, mp.current_process().name)
            break
        
        for row in content:
            writer.writerow(row)
        if count % 10 == 0 and count != 0:
            print '%d images have been collected in %s.' % (count, outFilePath)
            
    outcsv.flush()
    outcsv.close()

def cloudWorker(args):
    
    key, q_result, q_error = args
    
    isValid, syffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:
            # Load Image
            img = CloudImageLoader.keyToValidImage(key)
        
            imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
#             q_result.put((imData,imDim))
            X = FD.extractSingleImageFeatures(imData, 1)
            y_pred, y_proba = Clf.predict(X)
            result = zip([key.name], y_pred, y_proba, [imDim[0]], [imDim[1]])
            q_result.put(result)
#             q_result.put([key.name, y_pred, y_proba])
        except:
            q_error.put(zip([key.name], [key.size]))

def cloudWorkerPlus(args):
    
    key, q_result, q_error, q_invalid = args
    
    isValid, suffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:
            # Load Image
            img = CloudImageLoader.keyToValidImage(key)
            
            imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
#             q_result.put((imData,imDim))
            X = FD.extractSingleImageFeatures(imData, 1)
            y_pred, y_proba = Clf.predict(X)
            
#             c_class, c_probs = CDetector.getClassAndProabability(img)
#             composite_proba = 1
            
            result = zip([key.name], y_pred, y_proba, [imageFormat], [imDim[0]], [imDim[1]], [key.size])
            print result
            q_result.put(result)
#             q_result.put([key.name, y_pred, y_proba])
        except:
            q_error.put(zip([key.name], [key.size]))
            
    else:
        q_invalid.put(zip([key.name], [key.size]))


def cloudFileTransferWorker(args):
    
    key, q_result, q_error= args
    
    isValid, suffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is converting by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:
            # Load Image
            img = CloudImageLoader.keyToValidImage(key)
            
            dim = img.shape
            finalDim = 1280
            if dim[0] > finalDim or dim[1] > finalDim:
                img = ImageLoader.resizeConstantARWithNoEmpty(img, (finalDim, finalDim))
            
            tmpFileName = process_name + '.jpg'
            tmpFileName = os.path.join('temp', tmpFileName)

            cv.imwrite(tmpFileName, img, [cv.IMWRITE_JPEG_QUALITY, 90])
            newKeyName = key.name[0:(-len(suffix)-1)] + '&copy.jpg'
            cIL.upLoadingFile(newKeyName, tmpFileName)
            
#             os.remove(tmpFileName)
            newKey = cIL.bucket.get_key(newKeyName)
            result = zip([key.name], [key.size], [newKey.size])
            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size]))         

def getClassesOfSubFigure(img):
    # Dismantle image
    nodeList = Dmtler.dismantle(img)
            
    # Get number of sub-images
    if len(nodeList) > 0:
        numSubImages = len(nodeList)
        nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
    else:
        numSubImages = 1
        
    # Remove surrounding empty space
    nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
    # Load all sub-images
    if len(nodeList) > 0:
        imData, imDims, dimSum = ImageLoader.loadSubImagesByNodeList(img, nodeList, Opt.finalDim, Opt.preserveAspectRatio)
    else: 
        imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
    
    # Extracting Features
    X = FD.extractFeatures(imData, 1)
    # Classify
    y_pred, y_proba = Clf.predict(X)
    
    sub_classes = ''
    sub_probs = ''
    sub_height = ''
    sub_width = ''
    segmentations = ''
    for i in range(0, len(nodeList)):
        if i != 0:
            sub_classes += ','
            sub_probs += ','
            sub_height += ','
            sub_width += ','
            segmentations += ','
            
        sub_classes += str(y_pred[i])
        sub_height += str(imDims[i][0])
        sub_width += str(imDims[i][1])
        
        sub_prob = ''
        for j in range(0, len(y_proba[i])):
            if j != 0:
                sub_prob += ':'
            sub_prob += str(y_proba[i][j])
        
        sub_probs += sub_prob
        
        node = nodeList[i]
        segmentation = str(node.info['start'][0]) + ':' + \
                        str(node.info['start'][1]) + ':' + \
                        str(node.info['end'][0]) + ':' + \
                        str(node.info['end'][1])
        segmentations += segmentation
    return numSubImages, sub_classes, sub_probs, segmentations, sub_height, sub_width

        

def cloudDismentleWorker(args):
    
    key, q_result, q_error = args
    
    isValid, suffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:

            # Load image
            img = CloudImageLoader.keyToValidImage(key)
            
            # Dismantle image
            nodeList = Dmtler.dismantle(img)
            
            # Get number of sub-images
            if len(nodeList) > 0:
                numSubImages = len(nodeList)
                nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
            else:
                numSubImages = 1
            
            # Remove surrounding empty space
            nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
            # Load all sub-images
            if len(nodeList) > 0:
                imData, imDims, dimSum = ImageLoader.loadSubImagesByNodeList(img, nodeList, Opt.finalDim, Opt.preserveAspectRatio)
            else: 
                imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
            
            # Extracting Features
            X = FD.extractFeatures(imData, 1)
            # Classify
            y_pred, y_proba = Clf.predict(X)
    
            sub_classes = ''
            sub_probs = ''
            sub_height = ''
            sub_width = ''
            segmentations = ''
            for i in range(0, len(nodeList)):
                if i != 0:
                    sub_classes += ','
                    sub_probs += ','
                    sub_height += ','
                    sub_width += ','
                    segmentations += ','
                    
                sub_classes += str(y_pred[i])
                sub_height += str(imDims[i][0])
                sub_width += str(imDims[i][1])
                
                sub_prob = ''
                for j in range(0, len(y_proba[i])):
                    if j != 0:
                        sub_prob += ':'
                    sub_prob += str(y_proba[i][j])
                
                sub_probs += sub_prob
                
                node = nodeList[i]
                segmentation = str(node.info['start'][0]) + ':' + \
                                str(node.info['start'][1]) + ':' + \
                                str(node.info['end'][0]) + ':' + \
                                str(node.info['end'][1])
                segmentations += segmentation
                
            result = zip([key.name], [numSubImages], [sub_classes], [sub_probs], [segmentations], [sub_height], [sub_width])
            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size]))


def cloudDismentleWorker2(args):
    
    key, q_result, q_error = args
    
    isValid, imageFormat = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:

            # Load image
            img = CloudImageLoader.keyToValidImage(key)
            
            # Dismantle image
            nodeList = Dmtler.dismantle(img)
            
            # Get number of sub-images
            if len(nodeList) > 0:
                numSubImages = len(nodeList)
                nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
            else:
                numSubImages = 1
            
            # Remove surrounding empty space
            nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
            # Load all sub-images
            if len(nodeList) > 0:
                imData, imDims, dimSum = ImageLoader.loadSubImagesByNodeList(img, nodeList, Opt.finalDim, Opt.preserveAspectRatio)
            else: 
                imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
            
            # Extracting Features
            X = FD.extractFeatures(imData, 1)
            # Classify
            y_pred, y_proba = Clf.predict(X)
          
            key_names = []
            sub_image_ids = []
            sub_classes = []
            sub_probs = []
            segmentations = []
            sub_heights = []
            sub_widths = []
            
            for i in range(0, len(nodeList)):
                
                
                node = nodeList[i]
                segmentation = str(node.info['start'][0]) + ':' + \
                                str(node.info['start'][1]) + ':' + \
                                str(node.info['end'][0]) + ':' + \
                                str(node.info['end'][1])
                sub_image_id = key.name[0:-len(imageFormat)-1] + '_' + str(i)
                
                
                key_names.append(key.name)
                sub_image_ids.append(sub_image_id)
                sub_classes.append(y_pred[i])
                sub_probs.append(y_proba[i])
                segmentations.append(segmentation)
                sub_heights.append(imDims[i][0])
                sub_widths.append(imDims[i][1])
            
            result = zip(key_names, sub_image_ids, sub_classes, sub_probs, segmentations, sub_heights, sub_widths)
            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size])) 


def cloudDismentleWorkerDB(args):
    
    row, q_result, q_error, q_invalid = args
    keyname = row[0]
    key = cIL.getKey(keyname)
    
    isValid, suffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:

            # Load image
            img = CloudImageLoader.keyToValidImage(key)
            
            # Dismantle image
            nodeList = Dmtler.dismantle(img)
            
            # Get number of sub-images
            if len(nodeList) > 0:
                numSubImages = len(nodeList)
                nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
            else:
                numSubImages = 1
            
            # Remove surrounding empty space
            nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
            # Load all sub-images
            if len(nodeList) > 0:
                imData, imDims, dimSum = ImageLoader.loadSubImagesByNodeList(img, nodeList, Opt.finalDim, Opt.preserveAspectRatio)
            else: 
                imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
            
            # Extracting Features
            X = FD.extractFeatures(imData, 1)
            # Classify
            y_pred, y_proba = Clf.predict(X)
          
            key_names = []
            sub_image_ids = []
            sub_classes = []
            sub_probs = []
            segmentations = []
            sub_heights = []
            sub_widths = []
            
            for i in range(0, len(nodeList)):
                
                
                node = nodeList[i]
                segmentation = str(node.info['start'][0]) + ':' + \
                                str(node.info['start'][1]) + ':' + \
                                str(node.info['end'][0]) + ':' + \
                                str(node.info['end'][1])
                sub_image_id = key.name[0:-len(imageFormat)-1] + '_composite_' + str(i)
                
                
                key_names.append(key.name)
                sub_image_ids.append(sub_image_id)
                sub_classes.append(y_pred[i])
                sub_probs.append(y_proba[i])
                segmentations.append(segmentation)
                sub_heights.append(imDims[i][0])
                sub_widths.append(imDims[i][1])
            
            result = zip(key_names, sub_image_ids, sub_classes, sub_probs, segmentations, sub_heights, sub_widths)
            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size])) 

def localWorker(args):
    
    filename, q_result, q_error = args
    
    fname, suffix = Common.getFileNameAndSuffix(filename)
    if suffix in Opt.validImageFormat:
        
        process_name = mp.current_process().name
        print '%s is classified by %s' %(filename, process_name) ####
    
        # Loading Image
        imData, imDims, dimSum = ImageLoader.loadImagesByList([filename], Opt.finalDim)
    
        # Extracting Features
        X = FD.extractSingleImageFeatures(imData, 1)
        # Classifying
        y_pred, y_proba = Clf.predict(X)
        # Write back to queue
        q_result.put(zip(fname, y_pred, y_proba, [imDims[0][0]], [imDims[0][1]]))
        
        
def cloudDimWorker(args):
    
    key, q_result, q_error = args
    
    isValid, suffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:
            # Load Image
            img = CloudImageLoader.keyToValidImage(key)
        
            imDim = img.shape
            del img
            result = zip([key.name], [imDim[0]], [imDim[1]])
            q_result.put(result)

        except:
            q_error.put(zip([key.name], [key.size]))
            
            
def cloudWorkerDB(args):
    
    row, q_result, q_error, q_invalid = args
    keyname = row[0]
    key = cIL.getKey(keyname)
    
    isValid, suffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:
            # Load Image
            img = CloudImageLoader.keyToValidImage(key)
            
            imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
#             q_result.put((imData,imDim))
            X = FD.extractSingleImageFeatures(imData, 1)
            y_pred, y_proba = Clf.predict(X)
            
#             c_class, c_probs = CDetector.getClassAndProabability(img)
#             composite_proba = 1
            
            result = zip([key.name], y_pred, y_proba, [imageFormat], [imDim[0]], [imDim[1]], [key.size])
            print result
            q_result.put(result)
#             q_result.put([key.name, y_pred, y_proba])
        except:
            q_error.put(zip([key.name], [key.size]))
            
    else:
        q_invalid.put(zip([key.name], [key.size]))
            
def cloudFileTransferWorkerDB(args):
    
    row, q_result, q_error= args
    keyname = row[0]
    key = cIL.getKey(keyname)
    
    isValid, suffix = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is converting by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:
            # Load Image
            img = CloudImageLoader.keyToValidImage(key)
            
            dim = img.shape
            finalDim = 1280
            if dim[0] > finalDim or dim[1] > finalDim:
                img = ImageLoader.resizeConstantARWithNoEmpty(img, (finalDim, finalDim))
            
            tmpFileName = process_name + '.jpg'
            tmpFileName = os.path.join('temp', tmpFileName)

            cv.imwrite(tmpFileName, img, [cv.IMWRITE_JPEG_QUALITY, 90])
            del img
            newKeyName = key.name[0:(-len(suffix)-1)] + '&copy.jpg'
            cIL.upLoadingFile(newKeyName, tmpFileName)
            
#             os.remove(tmpFileName)
            newKey = cIL.bucket.get_key(newKeyName)
            result = zip([key.name], [key.size], [newKey.size])
            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size])) 