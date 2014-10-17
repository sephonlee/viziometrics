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

def worker(Opt, FD, Clf, key, q):
    # Download file from key
#     print key.name
    process_name = mp.current_process().name
    print process_name
    fname = "%s_img.%s" % (process_name, getFileSuffix(key))
    fp = open(fname, "w")
    key.get_file(fp)
    fp.close()
    
    imData, imDims, dimSum = ImageLoader.loadImages([fname], Opt.finalDim)
    print imData
#     q.put((imData,imDims, dimSum))
    # Extracting Features
    X = FD.extractSingleImageFeatures(imData, 1)
    print X
    y_pred, y_proba = Clf.predict(X)
    print y_pred
    q.put(zip([key.name], y_pred, y_proba))
    
        
def listener(q, path, filename):
    filePath = os.path.join(path, filename) + '.csv'
    count = 0
    while True:
        outcsv = open(filePath, 'ab')
        writer = csv.writer(outcsv, dialect = 'excel')
#     while True:
#         print 'listeining'
        content = q.get()
        count += 1
#         print 'number imags= ', count
        if content == 'kill':
            print('All saved. Stop %s' % mp.current_process().name)
            break
        writer.writerow(content)
        outcsv.flush()
        outcsv.close()
    
    
#################################
Opt = Opt(isClassify = True)
FD = FeatureDescriptor(Opt.dicPath) 
Clf = SVMClassifier(Opt, clfPath = Opt.svmModelPath)
keyPath = Opt.keyPath
f = open(keyPath, 'r')
access_key = f.readline()[0:-1]
secret_key = f.readline()
conn = S3Connection(access_key, secret_key)
bucket = conn.get_bucket(Opt.host)
bucketList = bucket.list()
#####################################
 
csvSavingPath = Opt.resultPath
csvFilename = 'class_result' 
filePath = os.path.join(csvSavingPath, csvFilename) + '.csv' 
outcsv = open(filePath, 'wb')
writer = csv.writer(outcsv, dialect = 'excel')
header = ['file_path', 'class_name', 'probability']
writer.writerow(header)
outcsv.close()
  
  
# try:
#     ###############################
manager = mp.Manager()
q = manager.Queue() 
       
       
#     p = mp.Process(target=listener, args=(q,csvSavingPath, csvFilename))
#     p.start()
       
pool = mp.Pool(processes = 10)
    # watcher = pool.apply_async(listener, (q, csvSavingPath, csvFilename,))
       
       
      
# pool.map(worker, Opt, FD, Clf, [key for key in bucketList], q)
       
for key in bucketList:
    if isKeyValidImageFormat(Opt, key):
#         pool.apply_async(worker, args=(Opt, FD, Clf, key, q,))    
        pool.apply(worker, args =  (Opt, FD, Clf, key, q,))
#         time.sleep(5)
 
             
#             worker(Opt, FD, Clf, key, q)
    #         print 'qget', q.get()
      
#     print key_.name
#     pool.map(worker(Opt, FD, Clf, key_, q))
#          
q.put('kill')
pool.close()
pool.join()
#     p.join()
 
# except KeyboardInterrupt:
#     print 'got ^C while pool mapping, terminating the pool'
#     pool.terminate()
# #     p.terminate()
#     print 'pool is terminated'
# except Exception, e:
#     print 'got exception: %r, terminating the pool' % (e,)
#     pool.terminate()
# #     p.terminate()
#     print 'pool is terminated'
# finally:
#     print 'joining pool processes'
#     pool.join()
# #     p.join()
#     print 'join complete'
# print 'the end'




 
# def cube(x):
#     print mp.current_process().name
#     return x**3
#   
# pool = mp.Pool(processes=4)
# 
# # for i in range(1,100):
# #     print pool.apply(cube, (i,))
# 
# results = pool.map(cube, range(1,100))
# print(results)
