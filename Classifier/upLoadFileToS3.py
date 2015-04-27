# Upload files to s3 server

from boto.s3.connection import S3Connection
from boto.s3.key import Key

from Options import *
from DataManager import *

Opt = Option()
CIL = CloudImageLoader(Opt)

keyName = 'finalClass_10_31_2014.csv'
keyPath = 'classification'
filePath = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/finalClass.csv'
CIL.upLoadingFileToPath(keyName, keyPath, filePath)

# full_key_name = os.path.join(keyPath, keyName)
# print full_key_name
# CIL.keyToFile(full_key_name, 'test')


keyName = 'errorList_10_31_2014.csv'
keyPath = 'classification'
filePath = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/finalError.csv'
CIL.upLoadingFileToPath(keyName, keyPath, filePath)


