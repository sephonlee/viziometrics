import sys
sys.path.append("..")
from DataFileTool.DataFileTool import *
from LatestModels import *
import itertools
import multiprocessing as mp

def flattenKeyList(key_list):
    result_list = []
    for row in key_list:
        result_list.append([row[0]])
    return result_list

def removeSuccessfulOperationKey(key_list, keyname):
    key_index = key_list.index(keyname)
    del key_list[key_index]
    return key_list

def saveBadKeyList(key_list, outPath, allKeyCsvFilename):
    outFilePath = os.path.join(outPath, allKeyCsvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')


def listener(name, q, outPath, outFilename, key_list, allKeyCsvFilename):
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
            #Save bad key list
            if key_list is not None:
                saveBadKeyList(key_list, outPath, allKeyCsvFilename)
            costTime = time.time() - startTime
            print'%d images have been classified in %d sec. Stop Listener in %s\n' % (count - 1, costTime, mp.current_process().name)
            break
        
        for row in content:
            writer.writerow(row)
            if key_list is not None:
                removeSuccessfulOperationKey(key_list, row[2])
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
            # Load image
            img = CloudImageLoader.keyToValidImage(key)
            
            # Dismantle image
            nodeList = DMTLER.dismantle(img)
            
            # Get number of sub-images
            if len(nodeList) > 0:
                numSubImages = len(nodeList)
        #             nodeList = DMTLER.updateImageToEffectiveAreaFromNodeList(img, nodeList, OPT_DMTLER.thresholds)
            else:
                numSubImages = 1
            
            # Remove surrounding empty space
            nodeList = DMTLER.updateImageToEffectiveAreaFromNodeList(img, nodeList, OPT_DMTLER.thresholds)
            # Load all sub-images
            if len(nodeList) > 0:
                imData, imDims, dimSum = ImageLoader.loadSubImagesByNodeList(img, nodeList, OPT_CCLF.finalDim, OPT_CCLF.preserveAspectRatio)
            else: 
                imData, imDim = ImageLoader.preImageProcessing(img, OPT_CCLF.finalDim, OPT_CCLF.preserveAspectRatio)
            
            # Extracting Features
            X = FD.extractFeatures(imData, 1)
            # Classify
            y_pred, y_proba = CCLF.predict(X)
          
            image_ids = []
            paper_ids = []
            img_locs = []
            segmentations = []
            class_names = []
            class_probs = []
            img_formats = []
            img_heights = []
            img_widths = []
            key_sizes = []
        
            if isinstance(nodeList, list):
                for i in range(0, len(nodeList)):
                                    
                    node = nodeList[i]
                    segmentation = str(node.info['start'][0]) + ':' + \
                                    str(node.info['start'][1]) + ':' + \
                                    str(node.info['end'][0]) + ':' + \
                                    str(node.info['end'][1])
                    
                    image_id = key.name.split('/')[-1]
                    paper_id = image_id.split('_')[0][3:]
                    image_format = image_id.split('.')[-1].lower()
                    image_id = image_id[0:-len(image_format)-1] + '_composite_' + str(i) + '.' + image_format
                    
                    image_ids.append(image_id)
                    paper_ids.append(paper_id)
                    img_locs.append(key.name)
                    segmentations.append(segmentation)
                    class_names.append(y_pred[i])
                    class_probs.append(y_proba[i])
                    img_formats.append(image_format)
                    img_heights.append(imDims[i][0])
                    img_widths.append(imDims[i][1])
                    key_sizes.append(key.size)
            else:
                i = 0
                segmentation = str('0' + ':' + \
                                    '0' + ':' + \
                                    str(img.shape[0]) + ':' + \
                                    str(img.shape[1])
                                    )
                image_id = key.name.split('/')[-1]
                paper_id = image_id.split('_')[0][3:]
                image_format = image_id.split('.')[-1].lower()
                image_id = image_id[0:-len(image_format)-1] + '_composite_' + str(i) + '.' + image_format
                
                image_ids.append(image_id)
                paper_ids.append(paper_id)
                img_locs.append(key.name)
                segmentations.append(segmentation)
                class_names.append(y_pred[i])
                class_probs.append(y_proba[i])
                img_formats.append(image_format)
                img_heights.append(imDims[i][0])
                img_widths.append(imDims[i][1])
                key_sizes.append(key.size)
                
            result = zip(image_ids, paper_ids, img_locs, segmentations, class_names, class_probs, img_formats, img_heights, img_widths, key_sizes)
            print result
            q_result.put(result)
        except:
            q_error.put(zip([key.name], [key.size]))  
    

def classifySubFigure(query):
    
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

    # Collect keys
    print 'Collecting keys from "%s"' % query
    key_list = IDM.getKeynamesByQuery(query)
    flat_key_list = flattenKeyList(key_list)
    
    num_keys = len(key_list)
    
    endTime = time.time()
    print num_keys, 'keys were collected in ', endTime - startTime, 'sec'
    
    
    # Create Multiprocess Manager
    manager = mp.Manager()  
    output_id = hash(query) / 10000000000000
    
    
    # All Key Out
    header = ['img_loc']
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'composite_Allkey_parallel_%s' % output_id
    Common.saveCSV(csvSavingPath, csvFilename, content = flat_key_list, header = header, mode = 'wb', consoleOut = False)
    
    # BadKey Key Out
    header = ['img_loc']
    csvSavingPath = Class_Classifier_Opt.resultPath
    allKeyCsvFilename = 'composite_baBkey_parallel_%s' % output_id
    Common.saveCSV(csvSavingPath, allKeyCsvFilename, header = header, mode = 'wb', consoleOut = False)
    
    # Result Out
    header = ['img_id', 'pmcid', 'img_loc', 'segmentation', 'class_name', 'class_probability', 'img_format', 'img_height', 'img_width', 'key_size']    
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'composite_result_parallel_%s' % output_id
    DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_result = manager.Queue() 
    p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename, key_list, allKeyCsvFilename))
    p_result.start()
             
    # Error Out
    header = ['img_id', 'file_size']
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'composite_error_parallel_%s' % output_id
    Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_error = manager.Queue() 
    p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename, None, None))
    p_error.start()
     
    # Invalid Out
    header = ['img_id', 'file_size']
    csvSavingPath = Class_Classifier_Opt.resultPath
    csvFilename = 'composite_invalid_parallel_%s' % output_id
    Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    q_invalid = manager.Queue() 
    p_invalid = mp.Process(target = listener, args=('Result', q_invalid, csvSavingPath, csvFilename, None, None))
    p_invalid.start()
    
    
                 
    pool = mp.Pool(processes = mp.cpu_count() + 2)
    print 'CPU count: %d' % mp.cpu_count()
    
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
    
    #Save BadKeys
    outFilePath = os.path.join(csvSavingPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    for row in key_list:
            writer.writerow(row)
    
    outcsv.flush()
    outcsv.close()
    
    'All cloud images (%d) were classified' % num_keys

if __name__ == '__main__':
    
#     query = 'SELECT s3.img_loc FROM s3_readable_keys as s3, image_composite as ic WHERE s3.img_id = ic.img_id AND s3.is_readable is null AND s3.img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" AND (s3.img_format = "png" or s3.img_format = "jpg") AND ic.is_composite = 1 limit 200000 offset 1300000'
#     query = 'SELECT s3.img_loc FROM s3_readable_keys as s3, image_composite_temp as ic WHERE s3.img_id = ic.img_id AND s3.is_readable is null AND key_size > 0 AND s3.img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" AND (s3.img_format = "jpg" OR s3.img_format = "png") AND ic.is_composite = 1 limit 100000 offset 30000'
    query = 'SELECT img_loc FROM dismantle_error'
    classifySubFigure(query)
    