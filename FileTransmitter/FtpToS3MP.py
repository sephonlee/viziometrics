import sys
sys.path.append("..")
from Extractor.TarParser import *
from Extractor.Common import *
import itertools
import multiprocessing as mp
from MultiProcessingWorker import *


def Ftp2S3TransmissionFromFileList(start, end):
    
    list_ftpTarInfo = []

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

        
    ##### Output CSVs
    manager = mp.Manager()  
      
    # Figure Result Out
    header = ['file_path', 'article_citation']
    csvFilename = 'pubmed_s3_paper_file_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_fileOnS3 = manager.Queue() 
    p_fileOnS3 = mp.Process(target = listener, args=('UL_file', q_fileOnS3, csvSavingPath, csvFilename))
    p_fileOnS3.start()
      
    # q_ftp_error Out
    header = ['ftp_file_path', 'Article Citation']
    csvFilename = 'pubmed_ftp_error_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_ftp_error = manager.Queue() 
    p_ftp_error = mp.Process(target = listener, args=('Ftp_error', q_ftp_error, csvSavingPath, csvFilename))
    p_ftp_error.start()
      
    # q_upload_error Out
    header = ['ftp_file_path']
    csvFilename = 'pubmed_upload_error_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_upload_error = manager.Queue() 
    p_upload_error = mp.Process(target = listener, args=('S3_UL_error', q_upload_error, csvSavingPath, csvFilename))
    p_upload_error.start()
     
     
    time.sleep(3) # wait for listener set
    # Pooling
    print 'Start Pooling...'
    startTime = time.time() 
    pool = mp.Pool(processes = mp.cpu_count() + 2)
#     pool = mp.Pool(processes = 5 + 2)
    results = pool.map(TarTransmisionWorker, \
                       itertools.izip(list_ftpTarInfo
                                    , itertools.repeat(q_ftp_error)       \
                                    , itertools.repeat(q_fileOnS3)        \
                                    , itertools.repeat(q_upload_error)))  \
    # Terminate processes
    endTime = time.time()
    print 'All images were classified in', endTime - startTime, 'sec.\n'
     
    q_fileOnS3.put('kill')
    q_ftp_error.put('kill')
    q_upload_error.put('kill')
     
    pool.close()
    pool.join()
    p_fileOnS3.join()
    p_ftp_error.join()
    p_upload_error.join()
    
def Ftp2S3TransmissionDB(query):
    

    ##### Gather Data from file_list
    print 'Collect tarfile path on FTP from query "%s"' %(query)
    startTime = time.time()
    
    
    db_info = getDBInfoFromFile(DBInfoPath)
    db = loginDB(db_info)
    cursor = db.cursor()
    cursor.execute(query)
    list_ftpTarInfo = cursor.fetchall()
                        
    ##### Output CSVs
    manager = mp.Manager()  
    output_id = hash(query) / 10000000000000
    # Figure Result Out
    header = ['file_path', 'article_citation']
    csvFilename = 'pubmed_s3_paper_file_%d' % (output_id)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_fileOnS3 = manager.Queue() 
    p_fileOnS3 = mp.Process(target = listener, args=('UL_file', q_fileOnS3, csvSavingPath, csvFilename))
    p_fileOnS3.start()
      
    # q_ftp_error Out
    header = ['ftp_file_path', 'Article Citation']
    csvFilename = 'pubmed_ftp_error_%d' % (output_id)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_ftp_error = manager.Queue() 
    p_ftp_error = mp.Process(target = listener, args=('Ftp_error', q_ftp_error, csvSavingPath, csvFilename))
    p_ftp_error.start()
      
    # q_upload_error Out
    header = ['ftp_file_path']
    csvFilename = 'pubmed_upload_error_%d' % (output_id)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_upload_error = manager.Queue() 
    p_upload_error = mp.Process(target = listener, args=('S3_UL_error', q_upload_error, csvSavingPath, csvFilename))
    p_upload_error.start()
     
     
    time.sleep(3) # wait for listener set
    # Pooling
    print 'Start Pooling...'
    startTime = time.time() 
    pool = mp.Pool(processes = mp.cpu_count() + 2)
#     pool = mp.Pool(processes = 5 + 2)
    results = pool.map(TarTransmisionWorkerDB, \
                       itertools.izip(list_ftpTarInfo
                                    , itertools.repeat(q_ftp_error)       \
                                    , itertools.repeat(q_fileOnS3)        \
                                    , itertools.repeat(q_upload_error)))  \
    # Terminate processes
    endTime = time.time()
    print 'All images were classified in', endTime - startTime, 'sec.\n'
     
    q_fileOnS3.put('kill')
    q_ftp_error.put('kill')
    q_upload_error.put('kill')
     
    pool.close()
    pool.join()
    p_fileOnS3.join()
    p_ftp_error.join()
    p_upload_error.join()   

if __name__ == '__main__':  

    start = 0
    end = 3
#     Ftp2S3TransmissionFromFileList(start, end)
    query = 'SELECT * FROM tarfile LIMIT 3'
    Ftp2S3TransmissionDB(query)
