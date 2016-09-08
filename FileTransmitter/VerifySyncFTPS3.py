import sys
sys.path.append("..")
from Extractor.TarParser import *
from Extractor.Common import *
import time
        

# Return All keys of tarfiles on S3
def getAllTarFilesOnS3(start, end, bucketList, csvSavingPath, keyPath = None, host = None):
    
    startTime = time.time()
    header = ['file_path', 'format', 'file_size']
    csvFilename = 'tar_files_onS3'
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    outFilePath = os.path.join(csvSavingPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    
    formats = {}
    count = 0
    for key in bucketList:
        count += 1
        if start < count < end:
            format = key.name.split('.')[-1]
            format = str(format)
#             if format == 'gz':
            writer.writerow(zip([key.name], [format], [key.size])[0])
            if format in formats:
                formats[format] += 1 
            else:
                formats[format] = 1   
            print 'count: %d'% (count)
        elif count > end:
            break
    print "start from %d to %d" % (start, end)
    
# Given filelist.csv, output the ftp_path versus s3_path
def getS32FTPFileList(csvSavingPath):
    
    header = ['file_path', 'ftp_path']
    csvFilename = 'S32FTP'
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    outFilePath = os.path.join(csvSavingPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    
    ##### Traverse
    startTime = time.time()
    
    endPoint = 0
    with open(ftp_file_list_csv ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        i = 0
        for i, row in enumerate(reader):
            keyname = os.path.join(S3Dir, row[0].split('/')[-1])
            writer.writerow([keyname, row[0]])
            endPoint = i
            if i % 100 == 0:
                print i, 'files has been verified'
                outcsv.flush()
                 
    endTime = time.time()
    print endPoint, 'tarfile paths were collected in ', endTime - startTime, 'sec'
    outcsv.flush()
    outcsv.close()
    
def getAllMissingTarFiles(csvSavingPath, bucketList):
        
    header = ['file_path']
    csvFilename = 'missing_tar_files_onS3'
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    outFilePath = os.path.join(csvSavingPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    
    ##### Traverse
    startTime = time.time()
     
    with open(ftp_file_list_csv ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        i = 0
        for i, row in enumerate(reader):
            keyname = os.path.join('tarfiles', row[0].split('/')[-1])
            key = bucket.get_key(keyname)
            if key is None:
                print row[0]
                print 'key not exist'
                writer.writerow([row[0]])
                outcsv.flush()
            dataEnd = True
            endPoint = i
            if i % 100 == 0:
                print i, 'files has been verified'

    endTime = time.time()
    print end - start, 'tarfile paths were collected in ', endTime - startTime, 'sec'
    outcsv.flush()
    outcsv.close()
#     for f in list_ftpTarInfo:
#         print f 

if __name__ == '__main__':  

    start = 0
    end = 7000000
    bucketList = bucket.list(prefix = 'tarfiles/')

    getAllTarFilesOnS3(start, end, bucketList, csvSavingPath, keyPath = None, host = None)
    
#     getAllMissingTarFiles(csvSavingPath, bucketList)
#     getS32FTPFileList(csvSavingPath)