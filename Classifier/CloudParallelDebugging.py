from boto.s3.connection import S3Connection
from boto.s3.key import Key
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools


from Dictionary import *
from Options import *
from DataManager import *
from Models import *

import multiprocessing as mp
# from MultiProcessingFunctions import *


def isKeyValidImageFormat(Opt, key):
    keyname =  key.name.split('.')
    if len(keyname) == 2:
        return keyname[1] in Opt.validImageFormat
    else:
        return False

def saveCSV(path, filename, content = None, header = None, mode = 'wb', consoleOut = True):

    if consoleOut:
        print 'Saving image information...'
    filePath = os.path.join(path, filename) + '.csv'
    with open(filePath, mode) as outcsv:
        writer = csv.writer(outcsv, dialect='excel')
        if header is not None:
            writer.writerow(header)
        if content is not None:
            for c in content:
                writer.writerow(c)
        
    if consoleOut:  
        print filename, 'were saved in', filePath, '\n'


def getFileSuffix(key):
    keyname =  key.name.split('.')
    if len(keyname) == 2:
        return keyname[1]
    else:
        return ''

    
#################################
Opt = Option(isClassify = True)
FD = FeatureDescriptor(Opt.dicPath) 
Clf = SVMClassifier(Opt, clfPath = Opt.svmModelPath)
cIL = CloudImageLoader(Opt)

keyPath = Opt.keyPath
f = open(keyPath, 'r')
access_key = f.readline()[0:-1]
secret_key = f.readline()
conn = S3Connection(access_key, secret_key)
bucket = conn.get_bucket(Opt.host)
bucketList = bucket.list()
#####################################
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
#             if imageFormat in ['tif', 'tiff']:
#                 img = CloudImageLoader.keyToValidImageOnDisk(key, process_name)
#             else:
            img = CloudImageLoader.keyToValidImage(key)
        
            imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim)
            q_result.put((imData,imDim))
#             X = FD.extractSingleImageFeatures(imData, 1)
#             y_pred, y_proba = Clf.predict(X)
#             q_result.put(zip([key.name], y_pred, y_proba))
        except:
            q_error.put([key.name, key.size])
    

def localWorker(Opt, FD, Clf, fname, q_result, q_error):
    process_name = mp.current_process().name
    print '%s is classified by %s' %(fname, process_name) ####

    # Loading Image
    imData, imDims, dimSum = ImageLoader.loadImagesByList([fname], Opt.finalDim)

#     q.put((imData,imDims, dimSum))
    # Extracting Features
    X = FD.extractFeatures(imData, 1)
    # Classifying
    y_pred, y_proba = Clf.predict(X)
    # Write back to queue
    q_result.put(zip([fname], y_pred, y_proba)) 

manager = mp.Manager()
# Result Out
header = ['file_path', 'class_name', 'probability']
csvSavingPath = Opt.resultPath
csvFilename = 'cloud_class_result_parallel'
Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
q_result = manager.Queue() 
p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
p_result.start()

# Error Out
header = ['file_path', 'file_size']
csvSavingPath = Opt.resultPath
csvFilename = 'cloud_error'
Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
q_error = manager.Queue() 
p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
p_error.start()
        
pool = mp.Pool(processes = 4)
print 'Pooling workers'
startTime = time.time()

start = 0
stop = 5

keys_to_process = [key for (i,key) in enumerate(bucketList) if start <= i and i < stop]
# print keys_to_process

test = [x for x in range(0,1000000000) if x<= 500 and x >= 0 else break]
print test

keys_to_process = []
 
# results = pool.map(cloudWorker,  bucketList)
 
for (i, key) in enumerate(bucketList):
    if i < 500:
        print i
        print key.name
        keys_to_process.append(key)
#         isValid, keyname = cIL.isKeyValidImageFormat(key)
#         if isValid:
#         pool.apply(cloudWorker, args = (key, q_result, q_error,))
    else:
        print keys_to_process
        endTime = time.time()
        print 'All images were classified in', endTime - startTime, 'sec'
        break
#     Terminate processes
# 
# startTime = time.time()     
# results = pool.map(cloudWorker, itertools.izip(keys_to_process, itertools.repeat(q_result), itertools.repeat(q_error)))
# endTime = time.time()
# print 'All images were classified in', endTime - startTime, 'sec'
# 
# # Terminate processes
# q_result.put('kill')
# q_error.put('kill')
# pool.close()
# pool.join()
# p_result.join()
# p_error.join()


 
       
# pool.map(worker, Opt, FD, Clf, [key for key in bucketList], q)
#         
# for key in bucketList:
#     if isKeyValidImageFormat(Opt, key):
# #         pool.apply_async(worker, args=(Opt, FD, Clf, key, q,))    
#         pool.apply(cloudWorker, args =  (Opt, FD, Clf, key, q,))
# #         time.sleep(5)
#   
#               
# #             worker(Opt, FD, Clf, key, q)
#     #         print 'qget', q.get()
#        
# #     print key_.name
# #     pool.map(worker(Opt, FD, Clf, key_, q))
# #          
# q.put('kill')
# pool.close()
# pool.join()
# p.join()
# 
# 
# 
# 
# def worker(Opt, FD, Clf, key, q):
#     # Download file from key
#     process_name = mp.current_process().name
#     print process_name
#     img =  CloudImageLoader.keyToValidImage(key)
#     plt.imshow(img)
#     plt.show()
#                     
#     print img.shape
#     fname = "%s_img.%s" % (process_name, getFileSuffix(key))
#     fp = open(fname, "w")
#     key.get_file(fp)
#     fp.close()
#     
#     imData, imDims, dimSum = ImageLoader.loadImagesByList([fname], Opt.finalDim)
# #     print imData
#     q.put((imData,imDims, dimSum))
#     # Extracting Features
# #     X = FD.extractSingleImageFeatures(imData, 1)
# #     print X
# #     y_pred, y_proba = Clf.predict(X)
# #     print y_pred
# #     q.put(zip([key.name], y_pred, y_proba))
#     
#         
# def listener(q, path, filename):
#     filePath = os.path.join(path, filename) + '.csv'
#     count = 0
#     while True:
#         outcsv = open(filePath, 'ab')
#         writer = csv.writer(outcsv, dialect = 'excel')
# #     while True:
#         content = q.get()
#         count += 1
# #         print 'number imags= ', count
#         if content == 'kill':
#             print('All saved. Stop %s' % mp.current_process().name)
#             break
#         writer.writerow(content)
#         outcsv.flush()
#         outcsv.close()
