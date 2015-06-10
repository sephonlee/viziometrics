import sys
sys.path.append("..")
from DataFileTool.DataFileTool import *
from LatestModels import *
import itertools
import multiprocessing as mp



def listener(name, q, outPath, outFilename):
    print '%s Listener set up in %s' % (name, mp.current_process().name)
    startTime = time.time()
    outFilePath = os.path.join(outPath, outFilename) + '.csv'
    count = 0
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    while True:
        content = q.get()
        if content is not None:
            count += 1
        # Stop
        if content == 'kill':
            costTime = time.time() - startTime
            print'%d images have been classified in %d sec. Stop Listener in %s\n' % (count - 1, costTime, mp.current_process().name)
            break
        
        for row in content:
            writer.writerow(row)
        if count % 10 == 0 and count != 0:
            print '%d images have been collected in %s.' % (count, outFilePath)
            
    outcsv.flush()
    outcsv.close()

def worker(args):
    
    row, q_result, q_error, q_invalid = args
    keyname = row[0]
    key = CIL.getKey(keyname)
    
    isValid, suffix = CIL.isKeyValidImageFormat(key)
    if isValid:
        process_name = mp.current_process().name
        print '%s (%d KB)is classified by %s' %(key.name, key.size, process_name) ####
        imageFormat = key.name.split('.')[-1]
        try:
            # Load Image
            img = CloudImageLoader.keyToValidImage(key)
            
            mask = DMTLER.getEffectiveRegionMask(img)
            classname, prob = CPSD.getClassAndProabability(mask)
            
            image_id = key.name.split('/')[-1]
            paper_id = image_id.split('_')[0][3:]
            
            is_composite = False
            if classname[0] == 'composite':
                is_composite = True
    
            result = zip([image_id], [paper_id], [key.name], [is_composite], prob)
            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size]))
            
    else:
        q_invalid.put(zip([key.name], [key.size]))


def classifyCompositeFigure(query):
    
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

    print 'Start classifying images on cloud server...'
    startTime = time.time()

    # Create Multiprocess Manager
    manager = mp.Manager()  
    output_id = hash(query) / 10000000000000
    
    # Result Out
    header = ['img_id', 'paper_id', 'img_loc', 'is_composite', 'probability']    
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'composite_result_parallel_%s' % output_id
    DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_result = manager.Queue() 
    p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
    p_result.start()
             
    # Error Out
    header = ['img_id', 'file_size']
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'composite_error_parallel_%s' % output_id
    Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_error = manager.Queue() 
    p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
    p_error.start()
     
    # Invalid Out
    header = ['img_id', 'file_size']
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'composite_invalid_parallel_%s' % output_id
    Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_invalid = manager.Queue() 
    p_invalid = mp.Process(target = listener, args=('Result', q_invalid, csvSavingPath, csvFilename))
    p_invalid.start()
                 
    pool = mp.Pool(processes = mp.cpu_count() + 2)
    print 'CPU count: %d' % mp.cpu_count()
    
    # Collect keys
    print 'Collecting keys from "%s"' % query
    key_list = IDM.getKeynamesByQuery(query)
    num_keys = len(key_list)
    endTime = time.time()
    print num_keys, 'keys were collected in ', endTime - startTime, 'sec'
    
    # Pooling
    print 'Start Pooling...'
    startTime = time.time() 
    results = pool.map(worker,  itertools.izip(key_list, itertools.repeat(q_result), itertools.repeat(q_error), itertools.repeat(q_invalid)))
    # Terminate processes
    endTime = time.time()
    print 'All images were classified in', endTime - startTime, 'sec.\n'
      
    q_result.put('kill')
    q_error.put('kill')
    q_invalid.put('kill')
    pool.close()
    pool.join()
    p_result.join()
    p_error.join()
    p_invalid.join()
    
    'All cloud images (%d) were classified' % num_keys

if __name__ == '__main__':
    
    query = "SELECT img_loc FROM s3_readable_keys WHERE is_readable is null AND img_format = 'jpg' limit 3000000 offset 200000"
    classifyCompositeFigure(query)
    