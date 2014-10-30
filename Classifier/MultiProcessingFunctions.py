import csv
import multiprocessing as mp
import os, errno
import time
from boto.s3.key import Key

from Dictionary import *
from Options import *
from DataManager import *
from Classification import *

## Global Object   
Opt = Option(isClassify = True)
FD = FeatureDescriptor(Opt.dicPath)
Clf = SVMClassifier(Opt, clfPath = Opt.svmModelPath)
cIL = CloudImageLoader(Opt)

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
        writer.writerow(content)
        if count % 10 == 0 and count != 0:
            print '%d images have been collected in %s.' % (count, outFilePath)
            
    outcsv.flush()
    outcsv.close()
    
def cloudWorker(args):
    
    key, q_result, q_error = args
    
    isValid, keyname = cIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[1]
        try:
            # Load Image
            if imageFormat in ['tif', 'tiff']:
                img = CloudImageLoader.keyToValidImageOnDisk(key, process_name)
            else:
                img = CloudImageLoader.keyToValidImage(key)
        
            imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim)
#             q_result.put((imData,imDim))
            X = FD.extractSingleImageFeatures(imData, 1)
            y_pred, y_proba = Clf.predict(X)
            result = zip([key.name], y_pred, y_proba)
            q_result.put(result)
#             q_result.put([key.name, y_pred, y_proba])
        except:
            q_error.put([key.name, key.size])
    

def localWorker(args):
    
    filename, q_result, q_error = args
    
    fname, suffix = Common.getFileNameAndSuffix(filename)
    if suffix in Opt.validImageFormat:
        
        process_name = mp.current_process().name
        print '%s is classified by %s' %(filename, process_name) ####
    
        # Loading Image
        imData, imDims, dimSum = ImageLoader.loadImagesByList([filename], Opt.finalDim)
    
#         q_result.put((imData,imDims, dimSum))
        # Extracting Features
        X = FD.extractSingleImageFeatures(imData, 1)
        # Classifying
        y_pred, y_proba = Clf.predict(X)
        # Write back to queue
        q_result.put(zip(fname, y_pred, y_proba))