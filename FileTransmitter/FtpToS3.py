# Upload file from FTP to S3
import sys
sys.path.append("..")
from Extractor.TarParser import *
from Extractor.Common import *


# TODO
def ftp2S3FromDB(DBInfoPath, query):
    
    db_info = getDBInfoFromFile(DBInfoPath)
    db = loginDB(db_info)
    cursor = db.cursor()
    cursor.execute(query)
    list_ftpTarInfo = cursor.fetchall()

    
    # Figure Result Out
    header = ['file_path']
    csvFilename = 'pubmed_s3_DB_paper_file'
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    outFilePath = os.path.join(csvSavingPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    
    total= len(list_ftpTarInfo)
    for (i, n) in enumerate(list_ftpTarInfo):
        f = n[1]
        print i+1, '/', total, 'filename:', f
        f = repr(f)[1:-3]
        ftpTarPath =  f
        tarName = ftpTarPath.split('/')[-1]
        local_tarPath = os.path.join(local_tmp_dir, tarName)
        print 'local_tarPath'
        try:
            download(ftpTarPath, local_tarPath)

            upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
            os.remove(local_tarPath)
             
            s3_fname = 'tarfiles/' + local_tarPath.split('/')[-1]
            writer.writerow([s3_fname])
            outcsv.flush()
        except:
            print f, 'is not existed on FTP'
        
    outcsv.flush()
    outcsv.close()

# TODO
def ftp2S3FromFileListViaDB(start, end, DBInfoPath, ftp_file_list_csv):

    db_info = getDBInfoFromFile(DBInfoPath)
    db = loginDB(db_info)
    cursor = db.cursor()
                       
    list_ftpTarInfo = []
#     list_localTarPath = []
    ##### Gather Data from file_list
    print 'Collect tarfile path on FTP from %d to %d...' %(start, end)
    startTime = time.time()
    
    with open(ftp_file_list_csv ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        i = 0
        for i, row in enumerate(reader):
            if i >= start and i < end:
                query = 'SELECT ftp_tar_loc FROM S32FTP WHERE tar_loc = "' + row[0] + '"' 
                cursor.execute(query)
                dbrow = cursor.fetchall()
#                 print repr(dbrow[0][0])[1:-3]
    
                list_ftpTarInfo.append([repr(dbrow[0][0])[1:-3]])
                dataEnd = True
                endPoint = i
            elif i >= end:
                dataEnd = False
                break
                
    endTime = time.time()
    print end - start, 'tarfile paths were collected in ', endTime - startTime, 'sec'
              
    # Figure Result Out
    header = ['file_path', 'article_citation']
    csvFilename = 'pubmed_s3_rest_paper_file_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    outFilePath = os.path.join(csvSavingPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    
    total= len(list_ftpTarInfo)
    for (i, f) in enumerate(list_ftpTarInfo):
        print i, '/', total, 'filename:', f
        ftpTarPath =  f[0]
        tarName = ftpTarPath.split('/')[-1]
#         print tarName
        local_tarPath = os.path.join(local_tmp_dir, tarName)
        try:
            download(ftpTarPath, local_tarPath)
        
            upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
#         os.remove(local_tarPath)
        
            writer.writerow([f[0]])
            outcsv.flush()
        except:
            print "the file is not available any more"
        
    outcsv.flush()
    outcsv.close()
    
def ftp2S3FromFileList(start, end, ftp_file_list_csv):
    
    list_ftpTarInfo = []
#     list_localTarPath = []
    ##### Gather Data from file_list
    print 'Collect tarfile path on FTP from %d to %d...' %(start, end)
    startTime = time.time()
    
    with open(ftp_file_list_csv ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        i = 0
        for i, row in enumerate(reader):
            if i >= start and i < end:
                list_ftpTarInfo.append(row)
                dataEnd = True
                endPoint = i
            elif i >= end:
                dataEnd = False
                break
                
    endTime = time.time()
    print end - start, 'tarfile paths were collected in ', endTime - startTime, 'sec'
              
    # Figure Result Out
    header = ['file_path', 'article_citation']
    csvFilename = 'pubmed_s3_rest_paper_file_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    outFilePath = os.path.join(csvSavingPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    
    total= len(list_ftpTarInfo)
    for (i, f) in enumerate(list_ftpTarInfo):
        print i, '/', total, 'filename:', f
        ftpTarPath =  f[0]
        tarName = ftpTarPath.split('/')[-1]
#         print tarName
        local_tarPath = os.path.join(local_tmp_dir, tarName)
#         download(ftpTarPath, local_tarPath)
        
        upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
        os.remove(local_tarPath)
        
        writer.writerow([f[0]])
        outcsv.flush()
        
    outcsv.flush()
    outcsv.close()
        

if __name__ == '__main__':  

    #837
    start = 1
    end = 2
    
    
    
    ftp_file_list_csv = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/caption/0506/paper_info_extraction/pubmed_extraction_error_0-900_.csv'
#     ftp_file_list_csv = '/home/ec2-user/VizioMetrics/class_result/ftp2S3/missing_tar_files_onS3.csv'
#     file_list = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/caption/finalExtractionError.csv'
#     file_list = '/home/ec2-user/VizioMetrics/class_result/ftp2S3/finalExtractionError.csv'
#     csvSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/file_transmission/pubmed_ftp'   
#     csvSavingPath = '/home/ec2-user/VizioMetrics/class_result/ftp2S3' 
        
#     ftp2S3FromFileList(start, end, ftp_file_list_csv)
    ftp2S3FromFileListViaDB(start, end, DBInfoPath, ftp_file_list_csv)
#     query = 'SELECT * FROM tarfile WHERE key_size = 0'
#     ftp2S3FromDB(DBInfoPath, query)