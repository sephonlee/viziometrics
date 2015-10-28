import sys
sys.path.append("..")

from Task.LatestModels import *
import csv
from random import randint

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





# scheme 630230
# visualization 455486
# photo 459321
# table 331883
# equation 1418169
# composite 1374721
# regexp for %copy 147385 / 15670

classname = 'visualization'

filepath = os.path.join('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Phylogenic')

os.mkdir(filepath)
# Result Out
header = ['img_loc', 'class_name', 'is_composite']    
csvSavingPath = filepath
csvFilename = classname
DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)


num_image = 1000
query_rand_1 = 'SELECT imgf.img_loc, imgf.class_name, imgc.is_composite' 
query_rand_2 = ' FROM image_full_info as imgf , image_composite as imgc WHERE imgf.img_id = imgc.img_id AND imgf.class_name = "%s" AND imgc.is_composite = 0 AND (imgf.img_format = "jpg" OR imgf.img_format = "png")' %(classname)
#     query_rand_3 = ' AND SUBSTRING_INDEX(imgf.img_id, ".", 1) not like "%copy"'
query_rand_3 = ' AND imgf.img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy"'
query_rand_4 = ' ORDER BY RAND() LIMIT %d' %num_image


query_rand = query_rand_1 + query_rand_2 + query_rand_3 + query_rand_4
print query_rand
key_list = IDM.getKeynamesByQuery(query_rand)

for keyname in key_list:
    print keyname[0]
    key = cImageLoader.getKey(keyname[0])
    localPath = os.path.join(filepath, key.name.split('/')[-1])
    print localPath

    outcsv = open(os.path.join(csvSavingPath, csvFilename + '.csv'), 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    writer.writerow(keyname)
    outcsv.flush()
    outcsv.close()
    key.get_contents_to_filename(localPath)
        