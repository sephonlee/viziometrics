from boto.s3.connection import S3Connection
from boto.s3.key import Key
import sys
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import csv
import itertools

from matplotlib import pyplot as plt
from Dictionary import *
from Options import *
from DataManager import *
from Models import *
from PIL import Image
from cStringIO import StringIO

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
        
            imData, imDim = ImageLoader.preImageProcessing(img, Opt.finalDim, Opt.preserveAspectRatio)
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
    imData, imDims, dimSum = ImageLoader.loadImagesByList([fname], Opt.finalDim, Opt.preserveAspectRatio)

#     q.put((imData,imDims, dimSum))
    # Extracting Features
    X = FD.extractFeatures(imData, 1)
    # Classifying
    y_pred, y_proba = Clf.predict(X)
    # Write back to queue
    q_result.put(zip([fname], y_pred, y_proba)) 

    def keyToValidImageOnDisk(key, filename): 

        keyname =  key.name.split('.')
        suffix = keyname[1]
        fname = filename + '.' + suffix
        fp = open(fname, "w")
        key.get_file(fp)
        fp.close()
        print 'error point'            
        img = cv.imread(fname, 0)
        return img
    
    def keyToValidImage(key):
        imgData = key.get_contents_as_string()
        fileImgData = StringIO(imgData)
        img = Image.open(fileImgData).convert('RGB')
        img = np.array(img) 
        if len(img.shape) == 3:
            img = img[:, :, ::-1].copy() 
        return img 




#### Test s3 image
# keyname = 'imgs/PMC3329770_a-68-00366-efd51.jpg'
# keyname = 'imgs/PMC3898118_onc2012600x1.tif'
# keyname = 'imgs/PMC2358978_pone.0002093.s003.tif'
# keyname = 'imgs/PMC1033567_medhist00152-0104.tif'
# keyname = 'imgs/PMC1033587_medhist00151-0069&copy.jpg'
keyname= 'tarfiles/Neurochem_Res_2009_Aug_28_34(8)_1522.tar.gz'
# keyname = 'imgs/PMC1994591_pone.0001006.s001.tif,527743'
key = bucket.get_key(keyname)
print key.size
# imgStringData = key.get_contents_as_string()
# cv
# nparr = np.fromstring(imgStringData, np.uint8)
# img_np = cv.imdecode(nparr, cv.CV_LOAD_IMAGE_COLOR)


# print cIL.isKeyValidImageFormat(key)
# 
# print key.size
# print img_np.shape
# print img_np.dtype
# plt.imshow(img_np, interpolation = 'bicubic')
# plt.show()
# cv.imwrite('/Users/sephon/Desktop/Research/VizioMetrics/test_ori.tif', img_np)
# 
# cv.imwrite('/Users/sephon/Desktop/Research/VizioMetrics/test_ori.jpg', img_np, [cv.IMWRITE_JPEG_QUALITY, 100])
# 
# img_np = ImageLoader.resizeConstantARWithNoEmpty(img_np, (1280, 1280))
# print img_np.shape
# cv.imwrite('/Users/sephon/Desktop/Research/VizioMetrics/test.jpg', img_np, [cv.IMWRITE_JPEG_QUALITY, 90])
# 
# 
# plt.imshow(img_np, interpolation = 'bicubic')
# plt.show()





### see error file
# finalClassFile = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/debug/error_filenames.csv'
# count = 0
# with open(finalClassFile ,'rb') as incsv:
#     reader = csv.reader(incsv, dialect='excel')
#     reader.next()
#     for row in reader:
#         if count > 1:
#         
#             keyname = row[0]
#             n = keyname.split('.')
# #             print n 
#             if n[1] in ['jpg', 'bmp', 'tif', 'tiff']:
#                 print row[0]
#                 key = bucket.get_key(keyname)
#                 print key.size
#                 imgStringData = key.get_contents_as_string()
#                 nparr = np.fromstring(imgStringData, np.uint8)
#                 img_np = cv.imdecode(nparr, cv.CV_LOAD_IMAGE_COLOR)
#                 
#     #             print img_np.shape
#                 plt.imshow(img_np, interpolation = 'bicubic')
#                 plt.show()
#         count += 1



# fd = cv.FeatureDetector_create('MSER')
# kpts = fd.detect(img_np)
# print 'mesr'
# print kpts
# print dir(kpts[0])
# print kpts[0].size
# print kpts[0].pt
# print kpts[0].octave
# print kpts[0].response
# 
# 
# _delta = 10 
# _min_area = 25 
# _max_area = 2000
# _max_variation = 10.0 
# _min_diversity = 10.0
# _max_evolution = 10 
# _area_threshold = 12.0
# _min_margin = 2.9 
# _edge_blur_size = 3 
# 
# mser = cv.MSER(10, 0, 100, 0.05, 0.02, 200, 1.01, 0.03, 7)
# gray = cv.cvtColor(img_np, cv.COLOR_BGR2GRAY)
# vis = img_np.copy()
# 
# regions = mser.detect(img_np)
# print regions[0]
# hulls = [cv.convexHull(p.reshape(-1, 1, 2)) for p in regions]
# cv.polylines(vis, hulls, 1, (0, 255, 0))
# 
# # cv.imshow('img', vis)
# plt.imshow(vis, interpolation = 'bicubic')
# plt.show()


# 
# im3 = img_np.copy()
# 
# gray = cv.cvtColor(img_np,cv.COLOR_BGR2GRAY)
# blur = cv.GaussianBlur(gray,(5,5),0)
# thresh = cv.adaptiveThreshold(blur,255,1,1,11,2)
# 
# #################      Now finding Contours         ###################
# 
# contours,hierarchy = cv.findContours(thresh,cv.RETR_LIST,cv.CHAIN_APPROX_SIMPLE)
# 
# samples =  np.empty((0,100))
# responses = []
# keys = [i for i in range(48,58)]
# 
# for cnt in contours:
#     if cv.contourArea(cnt)>50:
#         [x,y,w,h] = cv.boundingRect(cnt)
# 
#         if  h>28:
#             cv.rectangle(img_np,(x,y),(x+w,y+h),(0,0,255),2)
#             roi = thresh[y:y+h,x:x+w]
#             roismall = cv.resize(roi,(10,10))
#             cv.imshow('norm',img_np)
#             key = cv.waitKey(0)
# 
#             if key == 27:  # (escape to quit)
#                 sys.exit()
#             elif key in keys:
#                 responses.append(int(chr(key)))
#                 sample = roismall.reshape((1,100))
#                 samples = np.append(samples,sample,0)
# 
# responses = np.array(responses,np.float32)
# responses = responses.reshape((responses.size,1))
# print "training complete"




# manager = mp.Manager()
# # Result Out
# header = ['file_path', 'class_name', 'probability']
# csvSavingPath = Opt.resultPath
# csvFilename = 'cloud_class_result_parallel'
# Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
# q_result = manager.Queue() 
# p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
# p_result.start()
# 
# # Error Out
# header = ['file_path', 'file_size']
# csvSavingPath = Opt.resultPath
# csvFilename = 'cloud_error'
# Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
# q_error = manager.Queue() 
# p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
# p_error.start()
#         
# pool = mp.Pool(processes = 4)
# print 'Pooling workers'
# startTime = time.time()
# 
# start = 0
# stop = 5
# 
# keys_to_process = [key for (i,key) in enumerate(bucketList) if start <= i and i < stop]
# # print keys_to_process


