import sys
sys.path.append("..")
from Extractor.TarParser import *

from boto.s3.connection import S3Connection
from boto.s3.key import Key
from ftplib import FTP
import os
import csv
import time

def upLoadingFileToPath(bucket, keyName, keyPath, filePath):
    print 'Uploading %s as %s' %(filePath, keyName)
    full_key_name = os.path.join(keyPath, keyName)
    k = bucket.new_key(full_key_name)
    k.set_contents_from_filename(filePath)
    print 'Complete Uploading'

def csvTarSaver(csv_file_path, tar_file_data):
    outcsv = open(csv_file_path, 'ab')
    row = tar_file_data
    writer = csv.writer(outcsv, dialect = 'excel')
    result = (row[0], row[1])
    writer.writerow(result)
    outcsv.close()
    return


if __name__ == '__main__':  

    S3keyPath = '/Users/sephon/Desktop/Research/VizioMetrics/keys.txt'
#     S3keyPath = '/home/ec2-user/VizioMetrics/keys.txt'
    s3host = 'escience.washington.edu.viziometrics'
    
    
    f = open(S3keyPath, 'r')
    access_key = f.readline()[0:-1]
    secret_key = f.readline()
    conn = S3Connection(access_key, secret_key)
    bucket = conn.get_bucket(s3host)
    
    
    ftp = FTP('ftp.ncbi.nlm.nih.gov')     # connect to host, default port
    ftp.login()
    ftp.cwd('pub/pmc')   
#     filename = 'file_list.csv'
    filename = '/Users/sephon/Desktop/Research/VizioMetrics/file_transmission/pubmed_ftp/missing_tar_files_onS3.csv'
    
    S3Dir = 'tarfiles'              
    local_tmp_dir = 'tmp'
    csv_path ='csv'
    tar_csv = 'tar_file_list.csv'
#     if not os.path.exists(local_tmp_dir):
#         os.mkdir(local_tmp_dir, 0755);
#     if not os.path.exists(os.path.join(csv_path, tar_csv)):
#         os.mkdir(csv_path, 0755);
    
    local_tarPath = 'tmp/Acta_Crystallogr_Sect_E_Struct_Rep_Online_2012_Mar_24_68(Pt_4)_o1162.tar.gz'
    
#     ftp filename 06/c2/Acta_Crystallogr_Sect_E_Struct_Rep_Online_2012_Mar_24_68(Pt_4)_o1162.tar.gz
# local tmp/Acta_Crystallogr_Sect_E_Struct_Rep_Online_2012_Mar_24_68(Pt_4)_o1162.tar.gz
    
#     tarPath = '06/c2/Acta_Crystallogr_Sect_E_Struct_Rep_Online_2012_Mar_24_68(Pt_4)_o1162.tar.gz'
#     file = open(local_tarPath, 'wb')
#     ftp.retrbinary('RETR %s' % tarPath, file.write)
#     file.close()
#     print 'here'
    
    tarname = 'PLoS_One_2012_Apr_13_7(4)_e33042.tar.gz'
    keyname = os.path.join(S3Dir, tarname)
    local_tarPath = os.path.join(local_tmp_dir, tarname)
    key = bucket.get_key(keyname)
    key.get_contents_to_filename(local_tarPath)
    
    
    
    
#     outcsv = open(os.path.join(csv_path, tar_csv), 'wb')
#     writer = csv.writer(outcsv, dialect = 'excel')
#     header = ['title', 'article_citation']
#     writer.writerow(header)
#     outcsv.close()
#         
#         
#     tmp_save_path = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Source_paper/temp'
#     
#     
# #     local_tarPath = 'tmp/BMC_Immunol_2001_Jan_29_2_1.tar.gz'
# #     tarPath = '7b/6b/BMC_Immunol_2001_Jan_29_2_1.tar.gz'
# #     file = open(local_tarPath, 'wb')
# #     ftp.retrbinary('RETR %s' % tarPath, file.write)
# #     file.close()
#     
# #     local_tarPath = '/Users/sephon/Documents/workspace/VizClassification/FileTransmitter/tmp/Hist_Philos_Life_Sci_2015_Jan_8_36(3)_371-393.tar.gz'
# #     tarName = local_tarPath.split('/')[-1]
# #     print tarName
# #     upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
#     
#     
#     with open(filename ,'rb') as incsv:
#         reader = csv.reader(incsv, dialect='excel')
#         reader.next()
#         for row in reader:
#             # download file
#             tarPath =  row[0]
#             tarName = tarPath.split('/')[-1]
#             local_tarPath = os.path.join(local_tmp_dir, tarName)
#             file = open(local_tarPath, 'wb')
#             ftp.retrbinary('RETR %s' % tarPath, file.write)
#             file.close()
#              
# #             # Parse tar
# #             print local_tarPath
# #             paper_data, figure_data = TarParser().extractInfo(local_tarPath, tmp_save_path)
# #             print paper_data
# #             print figure_data
#              
#             # upload file
#             upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
#             #delete local file
#             os.remove(local_tarPath)
#             #save file info to csv
#             csvTarSaver(os.path.join(csv_path, tar_csv), [local_tarPath, row[1]])
# #             
#             