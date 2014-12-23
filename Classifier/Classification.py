# Visualization Classifier
# Main Program

from Dictionary import *
from Options import *
from DataManager import *
from Models import *
from MultiProcessingFunctions import *

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
            header = ['file_path', 'class_name', 'probability']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
            for key in bucketList:
                isValidImage, suffix = cImageLoader.isKeyValidImageFormat(key)
                if isValidImage:
                    
                    # Load Images
                    try:
                        img =  CloudImageLoader.keyToValidImage(key)
                    except:
                        img = CloudImageLoader.keyToValidImageOnDisk(key, 'tmp')
                        
                    imData, imDim = ImageLoader.preImageProcessing(img, self.Opt.finalDim)
                    
                    # Extracting Features
                    X = self.FeatureDescriptor.extractSingleImageFeatures(imData, 1)
                    # Classify
                    y_pred, y_proba = self.Classifier.predict(X)
                          
                    result = zip([key.name], y_pred, y_proba)
                    print result
                    Common.saveCSV(csvSavingPath, csvFilename, result, mode = 'ab', consoleOut = False)
                    nImageAll += 1 
                    if np.mod(nImageAll, 100) == 0:
                        print '%d images have been classified' % nImageAll  
            costTime = time.time() - startTime
            print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, os.path.join(csvSavingPath, csvFilename), costTime)
        else:
            print 'Classifier not loaded'         

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
            header = ['file_path', 'class_name', 'probability']
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
            print end - start, 'keys were classified in ', endTime - startTime, 'sec'
            print 'Collection ends at key index = %d' % i
            
            # Pooling
            print 'Start Pooling...'
            startTime = time.time() 
            results = pool.map(cloudWorker,  itertools.izip(keys_to_process, itertools.repeat(q_result), itertools.repeat(q_error)))
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
    
    def classifyLocalImagesParallel(self, corpusPath = None):
        if corpusPath is None:
            corpusPath = self.Opt.classifyCorpusPath
        
        if self.Classifier is not None:
            print 'Start classifying images parallel in local disk...'
            startTime = time.time()
            
            manager = mp.Manager()  
            # Result out
            header = ['file_path', 'class_name', 'probability']
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
            print len(file_to_process), 'keys were classified in ', endTime - startTime, 'sec'
            
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
            header = ['file_path', 'class_name', 'probability']
            csvSavingPath = self.Opt.resultPath
            csvFilename = 'class_result'
            Common.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
            
            for dirPath, dirNames, fileNames in os.walk(corpusPath):   
                for f in fileNames:
                    fname, suffix = Common.getFileNameAndSuffix(f)
                    if suffix in self.Opt.validImageFormat:
                        
                        filename = os.path.join(dirPath, f)
                        # Loading Images
                        imData, imDims, dimSum = ImageLoader.loadImagesByList([filename], self.Opt.finalDim)
                            
                        # Extracting Features
                        X = self.FeatureDescriptor.extractSingleImageFeatures(imData, 1)

                        # Classify
                        y_pred, y_proba = self.Classifier.predict(X)
                        
                        result = zip([filename], y_pred, y_proba)
                        Common.saveCSV(csvSavingPath, csvFilename, result, mode = 'ab', consoleOut = False)
                        nImageAll += 1 
                        if np.mod(nImageAll, 100) == 0:
                            print '%d images have been classified.' % nImageAll
            costTime = time.time() - startTime
            print 'All %d images were classified and saved in %s within %d sec.' % (nImageAll, os.path.join(csvSavingPath, csvFilename), costTime)
        else:
            print 'Classifier not loaded'         

if __name__ == '__main__': 
    
    Opt = Option(isClassify = True)
#     corpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/testCorpus"
    VCLF = VizClassifier(Opt, clf = 'SVM')
#     VCLF.classifyLocalImages()
#     VCLF.classifyCouldImages()
#     VCLF.classifyLocalImagesParallel()
    VCLF.classifyCloudImagesParallel(0, 100)
