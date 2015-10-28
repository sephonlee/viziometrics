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



# def updateImageToEffectiveArea(img, thresholds):
#     heads, ends = Dismantler.getEffectiveImageArea(img, thresholds)
#     
#     if ends[1] > heads[1] and ends[0] > heads[0]:
#         new_img = img[heads[1]:ends[1], heads[0]:ends[0]]
#     else:
#         new_img = img
#         
#     return new_img
#     
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
            imDim = img.shape
            # shrink to effective area
#             img = updateImageToEffectiveArea(img, OPT_DMTLER.thresholds)
            
            imData, imDim_shirnked = ImageLoader.preImageProcessing(img, OPT_CCLF.finalDim, OPT_CCLF.preserveAspectRatio)
            
            X = FD.extractSingleImageFeatures(imData, 1)
            y_pred, y_proba = CCLF.predict(X)
            
            image_id = key.name.split('/')[-1]
            pmcid = image_id.split('_')[0][3:]
            image_format = image_id.split('.')[-1].lower()
       
            result = zip([image_id], [pmcid], [key.name], y_pred, y_proba, [image_format], [imDim[0]], [imDim[1]], [key.size])

            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size]))
            
    else:
        q_invalid.put(zip([key.name], [key.size]))


def classifyCompositeFigure(query):
    
    if OPT_CCLF is not None:
        Class_Classifier_Opt = OPT_CCLF
    else:
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
    header = ['img_id', 'pmcid', 'img_loc', 'class_name', 'class_probability', 'img_format', 'img_height', 'img_width', 'key_size']    
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'class_result_parallel_%s' % output_id
    DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_result = manager.Queue() 
    p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
    p_result.start()
             
    # Error Out
    header = ['img_id', 'file_size']
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'class_error_parallel_%s' % output_id
    Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_error = manager.Queue() 
    p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
    p_error.start()
     
    # Invalid Out
    header = ['img_id', 'file_size']
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'class_invalid_parallel_%s' % output_id
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
    
    query = "SELECT img_loc FROM s3_readable_keys WHERE is_readable is null AND (img_format = 'jpg' OR img_format = 'png') limit 3000000 offset 0"
#     query = "select img_loc from keys_s3 WHERE img_format = 'jpg' LIMIT 1000000"
#     query = 'SELECT img_loc from s3_keys as s3 WHERE s3.img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" AND (s3.img_format = "jpg" OR s3.img_format = "png") AND img_id not in ( SELECT img_id from image_full_info WHERE img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" AND (img_format = "jpg" OR img_format = "png"))'

    classifyCompositeFigure(query)
    