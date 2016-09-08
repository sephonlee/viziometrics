from LatestModels import *
import csv
import cv2
from random import randint

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

Class_Classifier_Opt = Option(isClassify = True)
try:
    cImageLoader = CloudImageLoader(Class_Classifier_Opt)
    bucketList = cImageLoader.getBucketList()
except:
    print 'Unable to connect s3 server'

try:
    db_info = ImageDataManager.getDBInfoFromFile(Class_Classifier_Opt.DBInfoPath)
    db_info = { 'host': "54.213.68.209",
                'db_username': "poshen",
                'db_password': "escience",
                'db_name': "visio"}
    IDM = ImageDataManager(connectToDB = True, db_info = db_info)
except:
    print "Unable to connect SQL server"





# scheme 630230
# visualization 455486
# photo 459321
# table 331883
# equation 1418169
# composite 1374721
# regexp for %copy 147385 / 15670

num_image = 1400*1.14
classnames = ['table','equation']
# classnames = ['composite']

for classname in classnames:
    filepath = os.path.join('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/S3Sampling/sampling_sub_figures', classname)
    
    os.mkdir(filepath)
    # Result Out
    header = ['img_loc', 'class_name', 'segmentation', 'img_id']    
    csvSavingPath = filepath
    csvFilename = classname
    DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    query_1 = 'SELECT img_loc, class_name, segmentation, img_id FROM sub_image_full_info'
    query_2 = ' WHERE (img_format = "jpg" OR img_format = "png") AND class_name = "%s"' %classname
    query_3 = ' AND img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy"'
    query_4 = ' ORDER BY RAND() LIMIT %d' %num_image
    query_rand = query_1 + query_2 + query_3 + query_4
    
    print query_rand
    key_list = IDM.getKeynamesByQuery(query_rand)
    
    for keyname in key_list:
        print keyname[0]
        key = cImageLoader.getKey(keyname[0])
#         localPath = os.path.join(filepath, key.name.split('/')[-1])
        localPath = os.path.join(filepath, keyname[3])
        print localPath
        seg = keyname[2].split(":")
    
        outcsv = open(os.path.join(csvSavingPath, csvFilename + '.csv'), 'ab')
        writer = csv.writer(outcsv, dialect = 'excel')
        writer.writerow(keyname)
        outcsv.flush()
        outcsv.close()
        imgStringData = key.get_contents_as_string()

        nparr = np.fromstring(imgStringData, np.uint8)
        img_np = cv.imdecode(nparr, 1)
        
        segmentation = [keyname[2]]
        
        plt = showSegmentationByList(img_np, segmentation, show = False)
#         plt.show()
        plt.savefig(localPath, dpi = 300, format='png')
        plt.clf()
#         plt.close()

               
#         img_seg = img_np[int(seg[0]):int(seg[2]), int(seg[1]):int(seg[3])]
#         plt.imshow(img_seg, cmap = 'gray', interpolation = 'bicubic')
#         plt.show()
#         cv2.imwrite(localPath, img_seg)
#         key.get_contents_to_filename(localPath)
        
