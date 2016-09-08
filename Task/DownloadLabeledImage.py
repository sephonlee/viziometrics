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
    db_path = "/Users/sephon/Desktop/Research/VizioMetrics/Database_Information/viziometrix_db_info.txt"
    db_info = ImageDataManager.getDBInfoFromFile(db_path)
    IDM = ImageDataManager(connectToDB = True, db_info = db_info)
except:
    print "Unable to connect SQL server"


OUTPATH = os.path.join('/Users/sephon/Desktop/Research/VizioMetrics/Corpus/S3Sampling/sampling_labelled_image/')

if not os.path.isdir(OUTPATH):
    os.mkdir(OUTPATH);
    


# Result Out
header = ['img_loc', 'img_id', 'class_name', 'free_text_label']
item = ["'phylogenetic tree'", "'metabolic pathway'", "'electrophoresis gel'", "'gel'"]
csvSavingPath = OUTPATH
csvFilename = "image_metadata"
# DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)

 
query_1 = 'SELECT img_id, class_name, free_text_label FROM user_free_text_label'
query_2 = ' WHERE free_text_label IN (%s)'%(",".join(item))

query = query_1 + query_2
print query
key_list = IDM.getKeynamesByQuery(query)
 
for keyname in key_list:
    print "pubmed/img/" + keyname[0]
    key = cImageLoader.getKey("pubmed/img/" + keyname[0])
     
    new_class_name = '_'.join(keyname[2].split(" "))
    local_cat_path = os.path.join(OUTPATH, new_class_name)    
    if not os.path.isdir(local_cat_path):
        os.mkdir(local_cat_path); 
     
    localPath = os.path.join(local_cat_path, key.name.split('/')[-1])
    metadata = ["pubmed/img/" + keyname[0]] + [x for x in keyname]
 
    outcsv = open(os.path.join(csvSavingPath, csvFilename + '.csv'), 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    writer.writerow(metadata)
    outcsv.flush()
    outcsv.close()
    key.get_contents_to_filename(localPath)
         
 
    