import sys
sys.path.append("..")
from Extractor.TarParser import *

import itertools
import multiprocessing as mp
from Common import *
from MultiProcessingWorker import *

def extractTarInfoFromS3(start, end, csvSavingPath, bucketList, keyPath = None, host = None):
    
    startTime = time.time()
    ##### Output CSVs
    manager = mp.Manager()  
    # Paper Result Out
    header = ['pmcid', 'pmid', 'doi', 'longname', 'shortname', 'title', 'num_page', 'year_pub', 'month_pub', 'day_pub']
    csvFilename = 'pubmed_paper_info_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_paper_result = manager.Queue() 
    p_paper_result = mp.Process(target = listener, args=('Paper_Info', q_paper_result, csvSavingPath, csvFilename))
    p_paper_result.start()

    q_figure_result_list = []
    p_figure_result_list = []
    for i in range(0, 5):
        csvFilename = 'pubmed_figure_caption_%d-%d_%d' % (start, end, i)
        saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
        q_figure_result = manager.Queue()
        q_figure_result_list.append(q_figure_result)
        p_figure_result = mp.Process(target = listener, args=('Figure_Info', q_figure_result, csvSavingPath, csvFilename))
        p_figure_result_list.append(p_figure_result)
        p_figure_result_list[i].start()
      
      
    # q_extraction_error Out
    header = ['file_path', 'key_size'] 
    csvFilename = 'pubmed_extraction_error_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_extract_error = manager.Queue() 
    p_extract_error = mp.Process(target = listener, args=('extraction_error', q_extract_error, csvSavingPath, csvFilename))
    p_extract_error.start()
    
    time.sleep(1)

    print 'CPU count: %d' % mp.cpu_count()
    # Collect keys
    print 'Collect keys from %d to %d...' %(start, end)
    keys_to_process = []                
    endPoint = 0
    i = 0
    for (i, key) in enumerate(bucketList):
        if i >= start and i < end:
            keys_to_process.append(key)
            dataEnd = True
            print 'index = %d, key: %s' %(i, key.name)
            endPoint = i
        elif i >= end:
            dataEnd = False
            break
    endTime = time.time()
    print end - start, 'keys were collected in ', endTime - startTime, 'sec'
    print 'Collection ends at key index = %d' % i
    
    # Pooling
    pool = mp.Pool(processes = mp.cpu_count() + 2)
    print 'Start Pooling...'
    startTime = time.time() 
    results = pool.map(TarExtractingWorker,  itertools.izip(keys_to_process, itertools.repeat(q_paper_result), itertools.repeat(q_figure_result), itertools.repeat(q_extract_error)))
    # Terminate processes
    endTime = time.time()
    print 'All tarfiles were extracted in', endTime - startTime, 'sec.\n'
    
    q_paper_result.put('kill')
    q_extract_error.put('kill')
    for i in range(0, 5):
        q_figure_result_list[i].put('kill')
    
    pool.close()
    pool.join()
    p_paper_result.join()
    p_extract_error.join()
    for i in range(0, 5):
        p_figure_result_list[i].join()
    
    if dataEnd:
        print 'All cloud tarfiles (%d) were extracted' %endPoint



def extractTarInfoFromS3DB(query, DBInfoPath):
    
    db_info = getDBInfoFromFile(DBInfoPath)
    db = loginDB(db_info)
    
    startTime = time.time()
    
    ##### Output CSVs
    manager = mp.Manager()
    
    output_id = hash(query) / 10000000000000
    # Paper Result Out
    header = ['pmcid', 'pmid', 'doi', 'longname', 'shortname', 'title', 'num_page', 'year_pub', 'month_pub', 'day_pub']
    csvFilename = 'pubmed_paper_info_%d' % (output_id)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_paper_result = manager.Queue() 
    p_paper_result = mp.Process(target = listener, args=('Paper_Info', q_paper_result, csvSavingPath, csvFilename))
    p_paper_result.start()

    # Figure Result Out
    q_figure_result_list = []
    p_figure_result_list = []
    for i in range(0, 5):
        csvFilename = 'pubmed_figure_caption_%d_%d' % (output_id, i)
        saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
        q_figure_result = manager.Queue()
        q_figure_result_list.append(q_figure_result)
        p_figure_result = mp.Process(target = listener, args=('Figure_Info', q_figure_result, csvSavingPath, csvFilename))
        p_figure_result_list.append(p_figure_result)
        p_figure_result_list[i].start()
      
      
    # q_extraction_error Out
    header = ['file_path', 'key_size'] 
    csvFilename = 'pubmed_extraction_error_%d' % (output_id)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_extract_error = manager.Queue() 
    p_extract_error = mp.Process(target = listener, args=('extraction_error', q_extract_error, csvSavingPath, csvFilename))
    p_extract_error.start()
    
    # Get keys from database
    cursor = db.cursor()
    cursor.execute(query)
    keys_to_process = cursor.fetchall()
    num_keys = len(keys_to_process)
    print 'Collected %d keys...' %num_keys
    
    # Pooling
    print 'Start Pooling...'
    pool = mp.Pool(processes = mp.cpu_count() + 2)
    startTime = time.time() 
    results = pool.map(TarExtractingWorkerDB,  itertools.izip(keys_to_process, itertools.repeat(q_paper_result), itertools.repeat(q_figure_result_list), itertools.repeat(q_extract_error)))
    # Terminate processes
    endTime = time.time()
    print 'All tarfiles were extracted in', endTime - startTime, 'sec.\n'
    
    q_paper_result.put('kill')
    q_extract_error.put('kill')
    for i in range(0, 5):
        q_figure_result_list[i].put('kill')
     
    pool.close()
    pool.join()
    p_paper_result.join()
    p_extract_error.join()
    for i in range(0, 5):
        p_figure_result_list[i].join()
        

def extractTarInfoFromFileList(start, end, file_list, keyPath = None, host = None):
    
    startTime = time.time()
    ##### Output CSVs
    manager = mp.Manager()  
    # Paper Result Out
    header = ['pmcid', 'pmid', 'doi', 'longname', 'shortname', 'title', 'num_page', 'year_pub', 'month_pub', 'day_pub']
    csvFilename = 'pubmed_paper_info_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_paper_result = manager.Queue() 
    p_paper_result = mp.Process(target = listener, args=('Paper_Info', q_paper_result, csvSavingPath, csvFilename))
    p_paper_result.start()

    # Figure Result Out
    header = ['img_id', 'pmcid', 'caption']
    
    q_figure_result_list = []
    p_figure_result_list = []
    for i in range(0, 5):
        csvFilename = 'pubmed_figure_caption_%d-%d_%d' % (start, end, i)
        saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
        q_figure_result = manager.Queue()
        q_figure_result_list.append(q_figure_result)
        p_figure_result = mp.Process(target = listener, args=('Figure_Info', q_figure_result, csvSavingPath, csvFilename))
        p_figure_result_list.append(p_figure_result)
        p_figure_result_list[i].start()
      
      
    # q_extraction_error Out
    header = ['file_path', 'key_size'] 
    csvFilename = 'pubmed_extraction_error_%d-%d' % (start, end)
    saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_extract_error = manager.Queue() 
    p_extract_error = mp.Process(target = listener, args=('extraction_error', q_extract_error, csvSavingPath, csvFilename))
    p_extract_error.start()
    
    time.sleep(1)
#     pool = mp.Pool(processes = mp.cpu_count() + 2)
    pool = mp.Pool(processes = 10)
    print 'CPU count: %d' % mp.cpu_count()
    # Collect keys
    print 'Collect keys from %d to %d...' %(start, end)        
    endPoint = 0
    keys_to_process = []  
    with open(file_list ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        i = 0
        for i, row in enumerate(reader):
            if i >= start and i < end:
                key = bucket.get_key(row[0])
                print i, key.name, key.size
                keys_to_process.append(key)
                dataEnd = True
                endPoint = i
            elif i >= end:
                dataEnd = False
                break
        
        
    endTime = time.time()
    print end - start, 'keys were collected in ', endTime - startTime, 'sec'
    print 'Collection ends at key index = %d' % i
    
    # Pooling
    print 'Start Pooling...'
    startTime = time.time() 
    results = pool.map(TarExtractingWorker,  itertools.izip(keys_to_process, itertools.repeat(q_paper_result), itertools.repeat(q_figure_result_list), itertools.repeat(q_extract_error)))
    # Terminate processes
    endTime = time.time()
    print 'All tarfiles were extracted in', endTime - startTime, 'sec.\n'
    
    q_paper_result.put('kill')
    q_extract_error.put('kill')
    for i in range(0, 5):
        q_figure_result_list[i].put('kill')
    
    pool.close()
    pool.join()
    p_paper_result.join()
    p_extract_error.join()
    for i in range(0, 5):
        p_figure_result_list[i].join()
    
    
    if dataEnd:
        print 'All cloud tarfiles (%d) were classified' %endPoint
        
if __name__ == '__main__':  

    start = 0
    end = 880
    bucketList = bucket.list(prefix = 'tarfiles/')


#     extractTarInfoFromS3(start, end, csvSavingPath, bucketList)
    
#     file_list = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/caption/finalExtractionError.csv'
#     file_list = '/home/ec2-user/VizioMetrics/class_result/ftp2S3/finalExtractionError.csv' 
    file_list = '/home/ec2-user/VizioMetrics/paper_info_extraction/pubmed_extraction_error_0-900.csv' 
    extractTarInfoFromFileList(start, end, file_list)
    
#     query = 'SELECT tar_loc, ftp_tar_loc FROM tarfile LIMIT 100'
#     extractTarInfoFromS3DB(query, DBInfoPath)