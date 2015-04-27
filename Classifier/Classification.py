# Visualization Classifier
# Main Program
import sys
sys.path.append("..")

from Dictionary import *
from Options import *
from DataManager import *
from Models import *
from MultiProcessingFunctions import *
from Dismantler.Options_Dismantler import *
from Dismantler.Dismantler import * 
from DBManager import *

from sets import Set
import itertools
import multiprocessing as mp

class VizClassifier():
   
    Opt = None
    FeatureDescriptor = None
    Classifier = None
   
    def loadSVMClassifier(self):
        self.FeatureDescriptor = FeatureDescriptor(self.Opt.dicPath) 
        self.Classifier = SVMClassifier(self.Opt, clfPath = self.Opt.svmModelPath)
        print 'SVM Classifier ready. \n'
        
        
    def loadCNNlassifier(self):
        print 'CNN Classifier is not implemented. \n'
        
   
    def __init__(self, Opt, clf = 'SVM'):
        if Opt.isClassify:
            self.Opt = Opt
            if clf == 'SVM':
                self.loadSVMClassifier()
            elif clf == 'CNN':
                self.loadCNNClassifier()
            else:
                print clf, 'classifier is not valid'
        else:
            print 'Please change Options to classify mode (isClassify = True)'
        
    def getClouldImageDim(self, keyPath = None, host = None):
        if self.Classifier is not None:      
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            nImageAll = 0
            header = ['image_id', 'image_heigh', 'image_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
            for key in bucketList:
                isValidImage, suffix = cImageLoader.isKeyValidImageFormat(key)
                if isValidImage:
                    
                    # Load Images                 
                    if key.name.split('.')[-1] in ['tif', 'tiff']:
                        img = CloudImageLoader.keyToValidImageOnDisk(key, 'tmp')
                    else:
                        img =  CloudImageLoader.keyToValidImage(key)
                        
                    imDim = img.shape
                    del img
    
                    result = zip([key.name.split('/')[1]], [imDim[0]], [imDim[1]])
                    print result
                    Common.saveCSV(csvSavingPath, csvFilename, result, mode = 'ab', consoleOut = False)
                    nImageAll += 1 
                    if np.mod(nImageAll, 100) == 0:
                        print '%d images have been extracted' % nImageAll  
            costTime = time.time() - startTime
            print 'All %d images were extracted and saved in %s within %d sec.' % (nImageAll, os.path.join(csvSavingPath, csvFilename), costTime)
        else:
            print 'Classifier not loaded' 
        
    def classifyCouldImages(self, keyPath = None, host = None):
        
        if self.Classifier is not None:      
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            nImageAll = 0
            header = ['file_path', 'class_name', 'probability', 'image_height', 'image_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
            for key in bucketList:
                isValidImage, suffix = cImageLoader.isKeyValidImageFormat(key)
                if isValidImage:

                    img =  CloudImageLoader.keyToValidImage(key)
                        
                    imData, imDim = ImageLoader.preImageProcessing(img, self.Opt.finalDim, self.Opt.preserveAspectRatio)
                    
                    # Extracting Features
                    X = self.FeatureDescriptor.extractSingleImageFeatures(imData, 1)
                    # Classify
                    y_pred, y_proba = self.Classifier.predict(X)
                          
                    result = zip([key.name], y_pred, y_proba, [imDim[0]], [imDim[1]])
                    Common.saveCSV(csvSavingPath, csvFilename, result, mode = 'ab', consoleOut = False)
                    nImageAll += 1 
                    if np.mod(nImageAll, 100) == 0:
                        print '%d images have been classified' % nImageAll  
            costTime = time.time() - startTime
            print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, os.path.join(csvSavingPath, csvFilename), costTime)
        else:
            print 'Classifier not loaded'         

    def getCloudImagesDimParallel(self, start, end, keyPath = None, host = None):
        if self.Classifier is not None:
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            
            manager = mp.Manager()  
            # Result Out
            header = ['image_id', 'image_heigh', 'image_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_class_result_parallel_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
                        
            pool = mp.Pool(processes = mp.cpu_count() + 2)
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
                    print 'index = %d' %i
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
            results = pool.map(cloudDimWorker,  itertools.izip(keys_to_process, itertools.repeat(q_result), itertools.repeat(q_error)))
            # Terminate processes
            endTime = time.time()
            print 'All images were classified in', endTime - startTime, 'sec.\n'
            
            q_result.put('kill')
            q_error.put('kill')
            pool.close()
            pool.join()
            p_result.join()
            p_error.join()
            
            if dataEnd:
                print 'All cloud images (%d) were classified' %endPoint
           
           
    def convertCloudImageParallel(self, start, end, keyPath = None, host = None):
        if self.Classifier is not None:
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            
            manager = mp.Manager()
            
            # Result Out
            header = ['file_path', 'old_file_size', 'new_file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_converted_files_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
                        
            pool = mp.Pool(processes = mp.cpu_count() + 2)
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
            print 'Start Pooling...'
            startTime = time.time() 
            results = pool.map(cloudFileTransferWorker,  itertools.izip(keys_to_process, itertools.repeat(q_result), itertools.repeat(q_error)))
            # Terminate processes
            endTime = time.time()
            print 'All images were classified in', endTime - startTime, 'sec.\n'
            
            q_result.put('kill')
            q_error.put('kill')     
            pool.close()
            pool.join()
            p_result.join()
            p_error.join()
            
            if dataEnd:
                print 'All cloud images (%d) were classified' %endPoint     

    def classifyCloudImagesParallel(self, start, end, keyPath = None, host = None):
        if self.Classifier is not None:
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            
            manager = mp.Manager()  
            # Result Out
            header = ['file_path', 'class_name', 'probability', 'format', 'image_height', 'image_width', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_class_result_parallel_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
            
            # Invalid Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_invalid_result_parallel_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_invalid = manager.Queue() 
            p_invalid = mp.Process(target = listener, args=('Result', q_invalid, csvSavingPath, csvFilename))
            p_invalid.start()
                        
            pool = mp.Pool(processes = mp.cpu_count() + 2)
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
            print 'Start Pooling...'
            startTime = time.time() 
            results = pool.map(cloudWorkerPlus,  itertools.izip(keys_to_process, itertools.repeat(q_result), itertools.repeat(q_error), itertools.repeat(q_invalid)))
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
            
            if dataEnd:
                print 'All cloud images (%d) were classified' %endPoint
    
    def classifyLocalImagesParallel(self, corpusPath = None):
        if corpusPath is None:
            corpusPath = self.Opt.classifyCorpusPath
        
        if self.Classifier is not None:
            print 'Start classifying images parallel in local disk...'
            startTime = time.time()
            
            manager = mp.Manager()  
            # Result out
            header = ['file_path', 'class_name', 'probability', 'image_height', 'image_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'local_class_result_parallel'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
            
            pool = mp.Pool(processes = mp.cpu_count() + 2)
            
            # Collect filepath
            print 'Collect files...'
            file_to_process = []
            for dirPath, dirNames, fileNames in os.walk(corpusPath):   
                for f in fileNames:
                    file_to_process.append(os.path.join(dirPath, f))
            endTime = time.time()
            print len(file_to_process), 'fires were classified in ', endTime - startTime, 'sec'
            
            # Pooling
            print 'Start Pooling...'
            startTime = time.time() 
            results = pool.map(localWorker,  itertools.izip(file_to_process, itertools.repeat(q_result), itertools.repeat(q_error)))
            # Terminate processes
            endTime = time.time()
            print 'All images were classified in', endTime - startTime, 'sec.\n'
            
            # Terminate processes
            q_result.put('kill')
            q_error.put('kill')
            pool.close()
            pool.join()
            p_result.join()
            p_error.join()
             
    def classifyLocalImages(self, corpusPath = None):
        
        if corpusPath is None:
            corpusPath = self.Opt.classifyCorpusPath
        
        if self.Classifier is not None:
            print 'Start classifying images in local disk...'
            startTime = time.time()
            nImageAll = 0
            header = ['file_path', 'class_name', 'probability', 'image_height', 'image_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
            for dirPath, dirNames, fileNames in os.walk(corpusPath):   
                for f in fileNames:
                    fname, suffix = Common.getFileNameAndSuffix(f)
                    if suffix in self.Opt.validImageFormat:
                        
                        filename = os.path.join(dirPath, f)
                        # Loading Images
                        imData, imDims, dimSum = ImageLoader.loadImagesByList([filename], self.Opt.finalDim, self.Opt.preserveAspectRatio)
                            
                        # Extracting Features
                        X = self.FeatureDescriptor.extractSingleImageFeatures(imData, 1)

                        # Classify
                        y_pred, y_proba = self.Classifier.predict(X)
                        
                        result = zip([filename], y_pred, y_proba, [imDims[0][0]], [imDims[0][1]])
                        Common.saveCSV(csvSavingPath, csvFilename, result, mode = 'ab', consoleOut = False)
                        nImageAll += 1 
                        if np.mod(nImageAll, 100) == 0:
                            print '%d images have been classified.' % nImageAll
            costTime = time.time() - startTime
            print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, os.path.join(csvSavingPath, csvFilename), costTime)
        else:
            print 'Classifier not loaded'         


    def classifyCloudImagesWithDismantlerParallel(self, start, end, keyPath = None, host = None):
        if self.Classifier is not None:
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            
            manager = mp.Manager()  
            # Result Out
            header = ['file_path', 'sub_image_id', 'sub_class_name', 'sub_probability', 'segmentation', 'subimage_height', 'subimage_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_dismantle_result_parallel_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error_%d-%d' % (start,end)
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
                        
            pool = mp.Pool(processes = mp.cpu_count() + 2)
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
                    print 'index = %d' %i
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
            results = pool.map(cloudDismentleWorker2,  itertools.izip(keys_to_process, itertools.repeat(q_result), itertools.repeat(q_error)))
            # Terminate processes
            endTime = time.time()
            print 'All images were classified in', endTime - startTime, 'sec.\n'
            
            q_result.put('kill')
            q_error.put('kill')
            pool.close()
            pool.join()
            p_result.join()
            p_error.join()
            
            if dataEnd:
                print 'All cloud images (%d) were classified' %endPoint
           
    
    def classifyCloudImagesWithDismantler(self, keyPath = None, host = None):
        
        Opt_dmtler = Option_Dismantler(isTrain = False)
        Dmtler = Dismantler(Opt_dmtler)
        
        
        if self.Classifier is not None:      
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            nImageAll = 0
            header = ['file_path', 'number_subimages', 'sub_class_name', 'sub_probability', 'segmentation', 'subimage_height', 'subimage_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
            for key in bucketList:
                isValidImage, suffix = cImageLoader.isKeyValidImageFormat(key)
                if isValidImage:
                    
                    # Get image from s3
                    img =  CloudImageLoader.keyToValidImage(key)
                    
                    # Dismantle the image
                    nodeList = Dmtler.dismantle(img)
                        
                    # Get number of sub-images
                    if len(nodeList) > 0:
                        numSubImages = len(nodeList)
                    else:
                        numSubImages = 1
                        
                    # Remove surrounding empty space
                    nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
                    # Load all sub-images
                    if len(nodeList) > 0:
                        imData, imDims, dimSum = ImageLoader.loadSubImagesByNodeList(img, nodeList, self.Opt.finalDim, self.Opt.preserveAspectRatio)
                    else: 
                        imData, imDim = ImageLoader.preImageProcessing(img, self.Opt.finalDim, self.Opt.preserveAspectRatio)
                    
                    # Extracting Features
                    X = self.FeatureDescriptor.extractFeatures(imData, 1)

                    # Classify
                    y_pred, y_proba = self.Classifier.predict(X)
                    sub_classes = ''
                    sub_probs = ''
                    sub_height = ''
                    sub_width = ''
                    segmentations = ''
                    
                    for i in range(0, len(nodeList)):
                        if i != 0:
                            sub_classes += ','
                            sub_probs += ','
                            sub_height += ','
                            sub_width += ','
                            segmentations += ','
                            
                        sub_classes += str(y_pred[i])
                        sub_height += str(imDims[i][0])
                        sub_width += str(imDims[i][1])
                        
                        sub_prob = ''
                        for j in range(0, len(y_proba[i])):
                            if j != 0:
                                sub_prob += ':'
                            sub_prob += str(y_proba[i][j])
                        
                        sub_probs += sub_prob
                        
                        node = nodeList[i]
                        segmentation = str(node.info['start'][0]) + ':' + \
                                        str(node.info['start'][1]) + ':' + \
                                        str(node.info['end'][0]) + ':' + \
                                        str(node.info['end'][1])
                        segmentations += segmentation
                        
                    result = zip([key.name], [numSubImages], [sub_classes], [sub_probs], [segmentations], [sub_height], [sub_width])
                    Common.saveCSV(csvSavingPath, csvFilename, result, mode = 'ab', consoleOut = False)
                    nImageAll += 1 
                    if np.mod(nImageAll, 100) == 0:
                        print '%d images have been classified' % nImageAll  
            costTime = time.time() - startTime
            print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, os.path.join(csvSavingPath, csvFilename), costTime)
        else:
            print 'Classifier not loaded'         

        
    def classifyLocalImagesWithDismantler(self, corpusPath = None):
        
        Opt_dmtler = Option_Dismantler(isTrain = False)
        Dmtler = Dismantler(Opt_dmtler)
        
        if corpusPath is None:
            corpusPath = self.Opt.classifyCorpusPath
        
        if self.Classifier is not None:
            print 'Start classifying images with dismantler in local disk...'
            startTime = time.time()
            nImageAll = 0
            header = ['file_path', 'number_subimages', 'sub_class_name', 'sub_probability', 'segmentation', 'subimage_height', 'subimage_width']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
            for dirPath, dirNames, fileNames in os.walk(corpusPath):   
                for f in fileNames:
                    fname, suffix = Common.getFileNameAndSuffix(f)
                    if suffix in self.Opt.validImageFormat:
                        
                        filename = os.path.join(dirPath, f)
                        # Load image
                        img = ImageLoader.loadImageByPath(filename)
                        # Dismantle image
                        nodeList = Dmtler.dismantle(img)
                        
                        # Get number of sub-images
                        if len(nodeList) > 0:
                            numSubImages = len(nodeList)
                        else:
                            numSubImages = 1
                            
                        # Remove surrounding empty space
                        nodeList = Dmtler.updateImageToEffectiveAreaFromNodeList(img, nodeList, Opt_dmtler.thresholds)
                        # Load all sub-images
                        if len(nodeList) > 0:
                            imData, imDims, dimSum = ImageLoader.loadSubImagesByNodeList(img, nodeList, self.Opt.finalDim, self.Opt.preserveAspectRatio)
                        else: 
                            imData, imDims, dimSum = ImageLoader.loadImagesByList([filename], self.Opt.finalDim, self.Opt.preserveAspectRatio)
                        
                        # Extracting Features
                        X = self.FeatureDescriptor.extractFeatures(imData, 1)

                        # Classify
                        y_pred, y_proba = self.Classifier.predict(X)
                        
                        sub_classes = ''
                        sub_probs = ''
                        sub_height = ''
                        sub_width = ''
                        segmentations = ''
                        
                        for i in range(0, len(nodeList)):
                            if i != 0:
                                sub_classes += ','
                                sub_probs += ','
                                sub_height += ','
                                sub_width += ','
                                segmentations += ','
                                
                            sub_classes += str(y_pred[i])
                            sub_height += str(imDims[i][0])
                            sub_width += str(imDims[i][1])
                            
                            sub_prob = ''
                            for j in range(0, len(y_proba[i])):
                                if j != 0:
                                    sub_prob += ':'
                                sub_prob += str(y_proba[i][j])
                            
                            sub_probs += sub_prob
                            
                            node = nodeList[i]
                            segmentation = str(node.info['start'][0]) + ':' + \
                                            str(node.info['start'][1]) + ':' + \
                                            str(node.info['end'][0]) + ':' + \
                                            str(node.info['end'][1])
                            segmentations += segmentation
                            
                            

                        result = zip([filename], [numSubImages], [sub_classes], [sub_probs], [segmentations], [sub_height], [sub_width])
                        print result
                        Common.saveCSV(csvSavingPath, csvFilename, result, mode = 'ab', consoleOut = False)
                        nImageAll += 1 
                        if np.mod(nImageAll, 100) == 0:
                            print '%d images have been classified.' % nImageAll
            costTime = time.time() - startTime
            print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, os.path.join(csvSavingPath, csvFilename), costTime)
        else:
            print 'Classifier not loaded'         
  

    # Get all files in s3 and store in the csv
    def getAllFormat(self, start, end, keyPath = None, host = None):
        
        if self.Classifier is not None:      
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            nImageAll = 0
            header = ['file_path', 'format', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
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
                    writer.writerow(zip([key.name], [format], [key.size])[0])
                    if format in formats:
                        formats[format] += 1 
                    else:
                        formats[format] = 1
                    
                    print formats
                elif count > end:
                    break
            
            print "start from %d to %d" % (start, end)
    
    def convertCloudImageParallelDB(self, query, keyPath = None, host = None, DBInfoPath = None):
        if self.Classifier is not None:
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect cloud server'
            
            try:
                if DBInfoPath is None:
                    DBInfoPath = self.Opt.DBInfoPath
        
                db_info = ImageDataManager.getDBInfoFromFile(DBInfoPath)
                print db_info
                IDM = ImageDataManager(connectToDB = True, db_info = db_info)
            except:
                print "Unable to connect SQL server"
                
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            
            manager = mp.Manager()
            
            # Result Out
            output_id = hash(query) / 10000000000000
            
            header = ['file_path', 'old_file_size', 'new_file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_converted_files_%s' % output_id
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error_%s' % output_id
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
                        
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
            results = pool.map(cloudFileTransferWorkerDB,  itertools.izip(key_list, itertools.repeat(q_result), itertools.repeat(q_error)))
            # Terminate processes
            endTime = time.time()
            print 'All images were classified in', endTime - startTime, 'sec.\n'
            
            q_result.put('kill')
            q_error.put('kill')     
            pool.close()
            pool.join()
            p_result.join()
            p_error.join()
            
            print 'All cloud images (%d) were classified' %num_keys     
                
    def classifyCloudImagesParallelDB(self, query, keyPath = None, host = None, DBInfoPath = None):
        if self.Classifier is not None:
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect s3 server'
                
            try:
                if DBInfoPath is None:
                    DBInfoPath = self.Opt.DBInfoPath

                db_info = ImageDataManager.getDBInfoFromFile(DBInfoPath)
                
                IDM = ImageDataManager(connectToDB = True, db_info = db_info)
            except:
                print "Unable to connect SQL server"
            
            print 'Start classifying images on cloud server...'
            startTime = time.time()
            
            manager = mp.Manager()  
            
            output_id = hash(query) / 10000000000000

            # Result Out
            header = ['file_path', 'class_name', 'probability', 'format', 'image_height', 'image_width', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_class_result_parallel_%s' % output_id
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error_result_parallel_%s' % output_id
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
            
            # Invalid Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_invalid_result_parallel_%s' % output_id
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
            results = pool.map(cloudWorkerDB,  itertools.izip(key_list, itertools.repeat(q_result), itertools.repeat(q_error), itertools.repeat(q_invalid)))
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
            
    def classifyCloudSubImagesParallelDB(self, query, keyPath = None, host = None, DBInfoPath = None):
        if self.Classifier is not None:
            try:
                cImageLoader = CloudImageLoader(self.Opt, keyPath = keyPath, host = host)
                bucketList = cImageLoader.getBucketList()
            except:
                print 'Unable to connect s3 server'
                
            try:
                if DBInfoPath is None:
                    DBInfoPath = self.Opt.DBInfoPath
    
                db_info = ImageDataManager.getDBInfoFromFile(DBInfoPath)
                
                IDM = ImageDataManager(connectToDB = True, db_info = db_info)
            except:
                print "Unable to connect SQL server"
            
            print 'Start disantling and classifying images on cloud server...'
            startTime = time.time()
            
            manager = mp.Manager()  
            
            output_id = hash(query) / 10000000000000
    
            # Result Out
            header = ['file_path', 'number_subimages', 'sub_class_name', 'sub_probability', 'segmentation', 'subimage_height', 'subimage_width']    
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_subclass_result_parallel_%s' % output_id
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_result = manager.Queue() 
            p_result = mp.Process(target = listener, args=('Result', q_result, csvSavingPath, csvFilename))
            p_result.start()
            
            # Error Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_error_result_parallel_%s' % output_id
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            q_error = manager.Queue() 
            p_error = mp.Process(target = listener, args=('Error', q_error, csvSavingPath, csvFilename))
            p_error.start()
            
            # Invalid Out
            header = ['file_path', 'file_size']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'cloud_invalid_result_parallel_%s' % output_id
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
            results = pool.map(cloudDismentleWorkerDB,  itertools.izip(key_list, itertools.repeat(q_result), itertools.repeat(q_error), itertools.repeat(q_invalid)))
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
    
    Opt = Option(isClassify = True)
#     corpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/testCorpus"
    VCLF = VizClassifier(Opt, clf = 'SVM')
    print VCLF.Classifier.best_estimator_
    
#     VCLF.classifyLocalImages()
#     VCLF.classifyCouldImages()
#     VCLF.classifyLocalImagesParallel()
#     VCLF.classifyCloudImagesParallel(0, 10)
#     VCLF.getCloudImagesDimParallel(0, 100)
#     VCLF.classifyLocalImagesWithDismantler()
#     VCLF.classifyCloudImagesWithDismantler()
#     VCLF.classifyCloudImagesWithDismantlerParallel(0, 100)
#     VCLF.getAllFormat(0, 10000000)
#     VCLF.convertCloudImageParallel(0, 1000)
#     VCLF.classifyCloudImagesParallelDB(query)
    query = "select img_loc from keys_s3 WHERE img_format = 'jpg' LIMIT 1000000"
#     VCLF.classifyCloudImagesParallelDB(query)
#     VCLF.convertCloudImageParallelDB(query)
#     VCLF.convertCloudImageParallelDB(query)
#     VCLF.classifyCloudSubImagesParallelDB(query)
    

        