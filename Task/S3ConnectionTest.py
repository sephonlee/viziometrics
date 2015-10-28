from boto.s3.connection import S3Connection
import boto
import boto.s3.connection
from boto.s3.connection import OrdinaryCallingFormat
from boto.s3.key import Key
import cv2 as cv
import numpy as np


keyPath = '/Users/sephon/Desktop/Research/VizioMetrics/keys.txt'
f = open(keyPath, 'r')
access_key = f.readline()[0:-1]
secret_key = f.readline()
host_bucket = 'escience.washington.edu.viziometrics'

# conn = S3Connection(access_key, secret_key, calling_format = boto.s3.connection.OrdinaryCallingFormat)
conn = S3Connection(access_key, secret_key)
print access_key
print secret_key
print host_bucket


# https://s3-us-west-2.amazonaws.com/escience.washington.edu.viziometrics/pubmed/img/PMC100320_1471-2156-3-3-1.jpg

conn = boto.s3.connect_to_region(
   region_name = 'us-west-2',
   aws_access_key_id = access_key,
   aws_secret_access_key = secret_key,
   calling_format = boto.s3.connection.OrdinaryCallingFormat()
   )

bucket = conn.get_bucket(host_bucket)

# conn = boto.connect_s3(
#         aws_access_key_id = access_key,
#         aws_secret_access_key = secret_key,
# #         host = Opt.host,
#         is_secure=False,               # uncomment if you are not using ssl
#         calling_format = boto.s3.connection.OrdinaryCallingFormat(),
#         )
# # print conn
# print "here"
bucket = conn.get_bucket(host_bucket)
# bucket = conn.get_bucket(host_bucket, validate=False)
# # for bucket in conn.get_all_buckets():
# #     print "{name}\t{created}".format(
# #             name = bucket.name,
# #             created = bucket.creation_date,
# #     )
#     
# print bucket
bucketList = bucket.list()
print bucketList

# 
# #### Test s3 image
# # keyname = 'pubmed/img/PMC2195757_JEM991620.f1.jpg'
# 
#     
keyname = 'pubmed/img/PMC100320_1471-2156-3-3-1.jpg'
# # keyname = 'imgs/PMC2358978_pone.0002093.s003.tif'
# # keyname = 'imgs/PMC1033567_medhist00152-0104.tif'
# # keyname = 'imgs/PMC1033587_medhist00151-0069&copy.jpg'
# # keyname= 'tarfiles/Neurochem_Res_2009_Aug_28_34(8)_1522.tar.gz'
# # keyname = 'imgs/PMC1994591_pone.0001006.s001.tif,527743'
key = bucket.get_key(keyname)
print key
print key.name
print key.size
# imgStringData = key.get_contents_as_string()
# # cv
# nparr = np.fromstring(imgStringData, np.uint8)
# img_np = cv.imdecode(nparr, cv.CV_LOAD_IMAGE_COLOR)