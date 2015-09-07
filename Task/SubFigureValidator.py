from LatestModels import *
import csv
import cv2 as cv
from random import randint
import time

def showSegmentationByList(img, seg_list, show = True):
    
    plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
    coordinate = []
    for seg in seg_list:
        
        coordinates = seg.split(':')
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


Class_Classifier_Opt = Option(isClassify = True)
try:
    cImageLoader = CloudImageLoader(Class_Classifier_Opt)
    bucketList = cImageLoader.getBucketList()
except:
    print 'Unable to connect s3 server'

try:
    db_info = ImageDataManager.getDBInfoFromFile(Class_Classifier_Opt.DBInfoPath)
    IDM = ImageDataManager(connectToDB = True, db_info = db_info)
except:
    print "Unable to connect SQL server"
    
    
classname = 'sub_figure'
filepath = os.path.join('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/S3Sampling', classname)
# filepath = os.path.join('/home/ec2-user/VizioMetrics', classname)

# Result Out
header = ['img_loc', 'segmentation']    
csvSavingPath = filepath
csvFilename = classname
DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)

for i in range(0,100):
     
     
#     SUBSTRING_INDEX('dotblogs.com.tw', '.', 1)
     
    valid = False
    while not valid:
        #     offset = randint(0, 7644707)
        offset = randint(0, 1513962)
        query_regexp = 'SELECT imgf.img_loc FROM sub_image_full_info as imgf , image_composite as imgc WHERE imgf.img_id = imgc.img_id AND imgf.img_format = "jpg" AND SUBSTRING_INDEX(imgf.img_id, ".", 1) like "%copy"' + ' AND imgc.img_id REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" LIMIT 1 OFFSET %d' %offset
        query_1a = 'SELECT imgc.img_loc' 
        query_1b = 'SELECT count(*)'
        query_2 = ' FROM image_composite as imgc WHERE imgc.is_composite = 1 LIMIT 1 OFFSET %d' %(offset)
        query_A = query_1a + query_2 
        query_B = query_1b + query_2 
        print query_A
         
        key_list = IDM.getKeynamesByQuery(query_A)
        img_loc = key_list[0][0]
         
        print img_loc
        query_all_sub = 'SELECT imgf.img_loc, imgf.segmentation FROM sub_image_full_info as imgf WHERE imgf.img_loc = "%s"' %(img_loc)
        print query_all_sub
         
        key_list = IDM.getKeynamesByQuery(query_all_sub)
        print "%d sub-figures are collected" %len(key_list)
        valid = len(key_list) > 0
     
    print "Show segmentation..."
    segmentation = []
    key = None
    for keyname in key_list:
        key = cImageLoader.getKey(keyname[0])
        segmentation.append(keyname[1])
     
    print key.name, segmentation
         
    localPath = os.path.join(filepath, key.name.split('/')[-1])
     
 
    outcsv = open(os.path.join(csvSavingPath, csvFilename + '.csv'), 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    writer.writerow(keyname)
    outcsv.flush()
    outcsv.close()
     
 
#         key = cImageLoader.getKey(keyname)
    imgStringData = key.get_contents_as_string()
# cv
    nparr = np.fromstring(imgStringData, np.uint8)
    img_np = cv.imdecode(nparr, cv.CV_LOAD_IMAGE_COLOR)
#         key.get_contents_to_filename(localPath)
     
    plt = showSegmentationByList(img_np, segmentation, show = False)
     
    suffix = localPath.split('.')[-1]
#     localPath = localPath[0:-len(suffix)-1] + ".png"
    plt.savefig(localPath, format='jpg')
    plt.close()
    segmentation = None
#     plt.show()
    
    
#     plt.imshow(img_np, interpolation = 'bicubic')
#     plt.show()



 



print key_list