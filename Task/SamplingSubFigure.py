from LatestModels import *
import csv
import cv2 as cv
from random import randint
import time

def showSegmentationByList(img, seg_list, show = True):
    
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    coordinate = []
    for seg in seg_list:
        print seg
        coordinates = seg.split(':')
        print coordinates
        start = [coordinates[0], coordinates[1]]
        end = [coordinates[2], coordinates[3]]
        plt.plot([start[1], end[1]], [start[0], start[0]],'r')
        plt.plot([end[1], end[1]], [start[0], end[0]],'r')
        plt.plot([start[1], end[1]], [end[0], end[0]], 'r')
        plt.plot([start[1], start[1]], [start[0], end[0]], 'r')        
    
    dim = img.shape
    plt.axis([-10, dim[1] + 10, dim[0] + 10, -10])
    if show:
        plt.show()
    return plt


# keyname = 'pubmed/img/PMC2889767_zdb0071061950005.jpg'
# key = CIL.getKey(keyname)
#   
# imgStringData = key.get_contents_as_string()
# nparr = np.fromstring(imgStringData, np.uint8)
# img_np = cv.imdecode(nparr, cv.CV_LOAD_IMAGE_COLOR)
#   
# nodeList = DMTLER.dismantle(img_np)
# segmentations = []
#   
# for i in range(0, len(nodeList)):
#                       
#     node = nodeList[i]
#     segmentation = str(node.info['start'][0]) + ':' + \
#                     str(node.info['start'][1]) + ':' + \
#                     str(node.info['end'][0]) + ':' + \
#                     str(node.info['end'][1])
#     segmentations.append(segmentation)
#                       
# print segmentations


keyname = 'pubmed/img/PMC3995170_NIHMS560014-supplement-8.jpg'
segmentation = ['1949:0:2362:896', '0:0:455:473', '0:473:455:896', '455:0:873:475', '455:475:873:896', '873:0:1437:475', '873:475:1437:896', '1437:0:1949:473', '1437:473:1949:896']

key = CIL.getKey(keyname)
imgStringData = key.get_contents_as_string()
# cv
nparr = np.fromstring(imgStringData, np.uint8)
img_np = cv.imdecode(nparr, cv.CV_LOAD_IMAGE_COLOR)
plt = showSegmentationByList(img_np, segmentation, show = True)
plt.show()