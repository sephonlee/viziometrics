from boto.s3.connection import S3Connection
from boto.s3.key import Key
from PIL import Image
# import Image
# import Image
import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import matplotlib.cm as cm
from DataManager import *
from cStringIO import StringIO
import multiprocessing as mp

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
 

host = 'escience.washington.edu.viziometrics'  
keyPath = '/home/ec2-user/VizioMetrics/keys.txt'
f = open(keyPath, 'r')
access_key = f.readline()[0:-1]
secret_key = f.readline()
       
conn = S3Connection(access_key, secret_key)
bucket = conn.get_bucket(host)
bucketList = bucket.list()
  
 
### Test Read into Disk and open by openCV
# key = bucket.get_key('imgs/PMC100321_1471-2105-3-9-2.jpg') 
key = bucket.get_key('imgs/PMC100357_1471-2164-3-4-6.jpg.jpg')
img = keyToValidImageOnDisk(key, 'test2')
plt.imshow(img, cmap = cm.Greys_r)
plt.show()
  
  
# ### Test Read by stringIO
# imgData = key.get_contents_as_string()
# fileImgData = StringIO(imgData)
# img = Image.open(fileImgData).convert('RGB')
# img = np.asarray(img) 
# img = img[:, :, ::-1].copy()
# # plt.imshow(img, cmap = cm.Greys_r)
# # plt.show()
# 
# path_ = 'test2.tif'
# path = '/home/ec2-user/VizioMetrics/Code/test.tif'
# path2 = '/home/ec2-user/VizioMetrics/test.tif'
# 
# img = cv.imread(path_, 0)
# img_ = Image.open(path_).convert('RGB')



    
# import numpy as np
# 
# def im2col(Im, block, style='sliding'):
#     bx, by = block
#     Imx, Imy = Im.shape
#     colH = (Imx - bx + 1) * (Imx - bx + 1)
#     colW = bx * by 
#     imCol = np.zeros((colH, colW))
#     curCol = 0
#     for j in range(0, Imy):
#         for i in range(0, Imx):
#             if (i+bx <= Imx) and (j+by <= Imy):
#                 imCol[curCol, :] = Im[i:i+bx, j:j+by].T.reshape(bx*by)
#                 curCol += 1
#             else:
#                 break
#     return np.asmatrix(imCol)
# 
# 
# def subdivPooling(X, l):
#     n = np.min(X.shape[0:2])
#     split = int(round(float(n)/2))  
#     if l == 0:
#         Q = np.asmatrix(np.squeeze(np.sum(np.sum(X, axis = 0), axis = 0))) 
#         return Q.T
#     else:
#         nx, ny, nz = X.shape
#         Q = subdivPooling(X[0:split, 0:split, :], l-1)
#         Q = np.vstack((Q, subdivPooling(X[split:nx, 0:split, :], l-1)))
#         Q = np.vstack((Q, subdivPooling(X[0:split, split:ny, :], l-1)))
#         Q = np.vstack((Q, subdivPooling(X[split:nx, split:ny, :], l-1)))
#     return Q
# 
# 
# X = np.asmatrix (range(1,65))
# centroids  = np.reshape(np.asmatrix(range(22,62)),(10,4), 'F')
# finalDim = [8, 8, 1]
# M = np.asmatrix(range(8,12))
# P = np.reshape(np.asmatrix(range(11,27)),(4,4))
# subdivLevels = 1
# Nimages = X.shape[0]
# 
# ####################
# cc = np.asmatrix(np.sum(np.power(centroids,2), axis = 1).T)
# sz = finalDim[0] * finalDim[1]
# 
# 
# XC = np.zeros((Nimages, (4**subdivLevels)*Ncentroids))
# 
# 
# ps = im2col(np.reshape(X[i,0:sz], finalDim[0:2], 'F'), (rf, rf))
# 
# ps = np.divide(ps - np.mean(ps, axis = 1), np.asmatrix(range(1,50)).T)
# 
# ps = np.dot((ps - M), P)
# 
# xx = np.sum(np.power(ps, 2), axis = 1)
# 
# xc = np.dot(ps, centroids.T)
# 
# 
# 
# z = np.sqrt(cc + xx - 2*xc)
# 
# 
# v = np.min(z, axis = 1)
# 
# inds = np.argmin(z, axis = 1)
# 
# mu = np.mean(z, axis = 1)
# 
# 
# ps = mu - z
# ps[ps < 0] = 0
# 
# off = np.asmatrix(range(0, (z.shape[0])*Ncentroids, Ncentroids))
# 
# ps = np.zeros((ps.shape[0]*ps.shape[1],1))
# 
# ps[off.T + inds] = 1
# ps_ = np.reshape(ps, (z.shape[1],z.shape[0]), 'F').T#
# 
#     print 'final ps', ps
#                 
# prows = finalDim[0] - rf + 1
# pcols = finalDim[1]- rf + 1
# 
# ps_.reshape((prows, pcols, Ncentroids))
# 
# ps__ = np.reshape(ps_, (prows, pcols, Ncentroids), 'F')
# ps__.reshape((prows, pcols, Ncentroids))
#                 
# XC[i, :] = FeatureDescriptor.subdivPooling(ps, subdivLevels).T
#             
# print 'Extracting features:', i+1, '/', Nimages    
# endTime = time.time()
# print X.shape[0], 'feature vectors computed in', endTime-startTime, 'sec\n'
#             
# mean = np.mean(XC, axis = 0)
# #         print mean
# sd = np.sqrt(np.var(XC, axis = 0) + 0.01)
# #         print sd
# XCs = np.divide(XC - mean, sd)
# #       XCs = np.hstack([XCs, np.ones((XCs.shape[0],1))])
# 
# mean = np.asmatrix(np.mean(XC, axis = 0))
# #         print mean
# sd = np.sqrt(np.asmatrix(np.var(XC, axis = 0)) + 0.01)
