from boto.s3.connection import S3Connection
from boto.s3.key import Key
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import csv
# Viz Classifier
from Dictionary import *
from Options import *
from DataManager import *
from Classification import *

import multiprocessing as mp


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

def worker(Opt, FD, Clf, fname, q):
    process_name = mp.current_process().name
#     print process_name
    print fname
    
    imData, imDims, dimSum = ImageLoader.loadImagesByList([fname], Opt.finalDim)
    print imData
    q.put((imData,imDims, dimSum))
    # Extracting Features
#     X = FD.extractFeatures(imData, 1)
#     y_pred, y_proba = Clf.predict(X)
#     q.put(zip([fname], y_pred, y_proba))
    
        
def listener(q, path, filename):
    
    print 'Listen set up in', mp.current_process().name
    startTime = time.time()
    filePath = os.path.join(path, filename) + '.csv'
    count = 0
    while True:
        outcsv = open(filePath, 'ab')
        writer = csv.writer(outcsv, dialect = 'excel')
#     while True:
        print 'listeining'
        content = q.get()
        count += 1
#         print 'number imags= ', count
        if content == 'kill':
            costTime = time.time() - startTime
            print'%d images have been classified in %d sec. Stop %s\n' % (count - 1, costTime, mp.current_process().name)
            break
        writer.writerow(content)
        if count & 100 == 0:
            print '%d images have been classified.' % count
    
    outcsv.flush()
    outcsv.close()
    
    
#################################
Opt = Opt(isClassify = True)
FD = FeatureDescriptor(Opt.dicPath) 
Clf = SVMClassifier(Opt, clfPath = Opt.svmModelPath)

nImageAll = 0
header = ['file_path', 'class_name', 'probability']
csvSavingPath = Opt.resultPath
csvFilename = 'class_result_para'
Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)


manager = mp.Manager()
q = manager.Queue() 


p = mp.Process(target=listener, args=(q, csvSavingPath, csvFilename))
p.start()
       
pool = mp.Pool(processes = 4)
# pool.apply(listener, args=(q, csvSavingPath, csvFilename))

fileList = []
for dirPath, dirNames, fileNames in os.walk(Opt.classifyCorpusPath):
    for f in fileNames:
        fname, suffix = Common.getFileNameAndSuffix(f)
        if suffix in Opt.validImageFormat:
            filename = os.path.join(dirPath, f)
            pool.apply(worker, args =  (Opt, FD, Clf, filename, q))
#             pool.apply_async(worker, args =  (Opt, FD, Clf, filename, q))
#             worker(Opt, FD, Clf, filename, q)
            
# print fileList
#    
# from itertools import repeat         
# # zip(repeat(Opt), repeat(FD), repeat(Clf), fileList, repeat(q))
#             
#             
# pool.map(worker, zip(repeat(Opt), repeat(FD), repeat(Clf), fileList, repeat(q)))

q.put('kill')
pool.close()
pool.join()
p.join()