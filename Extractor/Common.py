import csv
import os
import time
from boto.s3.connection import S3Connection
from boto.s3.key import Key
from ftplib import FTP
import MySQLdb
from random import randint

#S3 Setting
S3keyPath = '/Users/sephon/Desktop/Research/VizioMetrics/keys.txt'
# S3keyPath = '/home/ec2-user/VizioMetrics/keys.txt'
s3host = 'escience.washington.edu.viziometrics'
f = open(S3keyPath, 'r')
access_key = f.readline()[0:-1]
secret_key = f.readline()
conn = S3Connection(access_key, secret_key)
bucket = conn.get_bucket(s3host)
S3Dir = 'tarfiles'  


# Local File Setting
ftp_file_list_csv = 'file_list.csv'            
local_tmp_dir = 'tmp'
local_tar_tmp_dir = 'tar_tmp'
csvSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/file_transmission/pubmed_ftp'
# csvSavingPath = '/home/ec2-user/VizioMetrics/class_result/ftp2S3' 
# csvSavingPath = '/home/ec2-user/VizioMetrics/paper_info_extraction' 

# DB Setting
# DBInfoPath = '/home/ec2-user/VizioMetrics/db_info.txt'
DBInfoPath = '/Users/sephon/Desktop/Research/VizioMetrics/Database_Information/support_db_info.txt'

def getDBInfoFromFile(path):
    f = open(path, 'r')
    db_info = { 'host': f.readline()[0:-1],
            'db_username': f.readline()[0:-1],
            'db_password': f.readline()[0:-1],
            'db_name': f.readline()[0:-1]}
    return db_info

def loginDB(db_info):
    host = db_info['host']
    db_username = db_info['db_username']
    db_password = db_info['db_password']
    db_name = db_info['db_name']
    db = MySQLdb.connect(host, db_username, db_password, db_name)
    return db

def download(ftpTarPath, local_tarPath):
    print 'Downloading %s from FTP to %s' %(ftpTarPath, local_tarPath) ####
    ftp = FTP('ftp.ncbi.nlm.nih.gov')     # connect to host, default port
    ftp.login()
    ftp.cwd('pub/pmc')
    file = open(local_tarPath, 'wb')
    ftp.retrbinary('RETR %s' % ftpTarPath, file.write)
    file.close()
    
def upLoadingFileToPath(bucket, keyName, keyPath, filePath):
    print 'Uploading %s as %s' %(filePath, keyName)
    full_key_name = os.path.join(keyPath, keyName)
    print full_key_name
    k = bucket.new_key(full_key_name)
    k.set_contents_from_filename(filePath)
    print 'Complete Uploading'
    
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