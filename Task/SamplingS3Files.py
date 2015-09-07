from LatestModels import *
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

classname = '2000'
filepath = os.path.join('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/S3Sampling', classname)

# Result Out
header = ['img_loc', 'class_name', 'is_composite']    
csvSavingPath = filepath
csvFilename = classname
DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)

for i in range(0,100):
    
    offset = randint(0, 44526)
    query_2000 = 'SELECT img_loc FROM image_full WHERE year_pub >= 1997 AND year_pub <= 2000 AND img_loc is not null AND img_format = "jpg" AND img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" LIMIT 1 OFFSET %d;' %offset
    query_regexp = 'SELECT imgf.img_loc, imgf.class_name, imgc.is_composite FROM image_full_info as imgf , image_composite as imgc WHERE imgf.img_id = imgc.img_id AND imgf.img_format = "jpg" AND SUBSTRING_INDEX(imgf.img_id, ".", 1) like "%copy"' + ' AND imgc.img_id REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" LIMIT 1 OFFSET %d' %offset
    query_1a = 'SELECT imgf.img_loc, imgf.class_name, imgc.is_composite' 
    query_1b = 'SELECT count(*)'
    query_2 = ' FROM image_full_info as imgf , image_composite as imgc WHERE imgf.img_id = imgc.img_id AND imgf.class_name = "%s" AND imgc.is_composite = 0 AND imgf.img_format = "jpg"' %(classname)
    query_3 = ' AND SUBSTRING_INDEX(imgf.img_id, ".", 1) not like "%copy"'
    query_4 = ' LIMIT 1 OFFSET %d' %offset
    query_A = query_1a + query_2 + query_3 + query_4
    query_B = query_1b + query_2 + query_3
    print query_2000
    
#     key_list = IDM.getKeynamesByQuery(query_2000)
# #     print key_list
# 
#     for keyname in key_list:
#         print keyname[0]
#         key = cImageLoader.getKey(keyname[0])
#         localPath = os.path.join(filepath, key.name.split('/')[-1])
#         print i, localPath
# 
#         outcsv = open(os.path.join(csvSavingPath, csvFilename + '.csv'), 'ab')
#         writer = csv.writer(outcsv, dialect = 'excel')
#         writer.writerow(keyname)
#         outcsv.flush()
#         outcsv.close()
# #     print key.name
#         key.get_contents_to_filename(localPath)

print key_list