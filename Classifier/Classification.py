# Viz Classifier
from Dictionary import *
from Options import *
from DataManager import *
# from MultiProcessingFunctions import *

# SVM Classifier
from sklearn.externals import joblib
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics
# from cStringIO import StringIO
import csv
import itertools
# from PIL import Image

import multiprocessing as mp


# This class needs a given path to dictionary
class FeatureDescriptor():
    
    def __init__(self, dicPath):
        
        try:
            self.dicPath = dicPath
            dicPath = os.path.join(dicPath, 'dictionaryModel.npz')
            dictionary = np.load(dicPath)
            print 'Dictionary loaded from', dicPath
            self.rf = dictionary['rfSize']
            self.finalDim = dictionary['finalDim']
            self.M = dictionary['Mean']
            self.P = dictionary['Patch']
            self.whitening = dictionary['whitening']
            self.centroids = dictionary['centroids']
            self.Ncentroids = dictionary['Ncentroids']
            self.FSMean = dictionary['FSMean']
            self.FSSd = dictionary['FSSd']
            
            if self.FSMean == 'unavailable':
                print 'Feature Descriptor is not available'
                self.unTrained = True
            else:
                self.unTrained = False
            
        except:
            print 'Unable to load dictionary'
        
    @ staticmethod
    def getFeatureNames(Ncentroids):
            
        featureNames = []
        for i in range(1, Ncentroids+1):
            name = 'centroids_' + str(i)
            featureNames.append(name)
        return np.asmatrix(featureNames).T
    
    @ staticmethod 
    def im2col(Im, block, style='sliding'):

        bx, by = block
        Imx, Imy = Im.shape
        colH = (Imx - bx + 1) * (Imx - bx + 1)
        colW = bx * by 
        imCol = np.zeros((colH, colW))
        curCol = 0
        for j in range(0, Imy):
            for i in range(0, Imx):
                if (i+bx <= Imx) and (j+by <= Imy):
                    imCol[curCol, :] = Im[i:i+bx, j:j+by].T.reshape(bx*by)
                    curCol += 1
                else:
                    break
        return np.asmatrix(imCol)
    
    @ staticmethod
    def subdivPooling(X, l):
        n = np.min(X.shape[0:2])
        split = int(round(float(n)/2))
        
        if l == 0:

            Q = np.asmatrix(np.squeeze(np.sum(np.sum(X, axis = 0), axis = 0)))
            
            return Q.T
    
        else:
            nx, ny, nz = X.shape
            Q = FeatureDescriptor.subdivPooling(X[0:split, 0:split, :], l-1)
            Q = np.vstack((Q, FeatureDescriptor.subdivPooling(X[split:nx, 0:split, :], l-1)))
            Q = np.vstack((Q, FeatureDescriptor.subdivPooling(X[0:split, split:ny, :], l-1)))
            Q = np.vstack((Q, FeatureDescriptor.subdivPooling(X[split:nx, split:ny, :], l-1)))
 
        return Q
    
    def extractSingleImageFeatures (self, X, subdivLevels = 1):
        
        if not self.unTrained:
            Nimages = 1
    
            cc = np.asmatrix(np.sum(np.power(self.centroids,2), axis = 1).T)
            sz = self.finalDim[0] * self.finalDim[1]
    
            XC = np.zeros((Nimages, (4**subdivLevels)*self.Ncentroids))
    
            ps = FeatureDescriptor.im2col(np.reshape(X[0,0:sz], self.finalDim[0:2], 'F'), (self.rf, self.rf))
            ps = np.divide(ps - np.mean(ps, axis = 1), np.sqrt(np.var(ps, axis = 1) +1))
    
            if self.whitening:
                ps = np.dot((ps - self.M), self.P)
                
            xx = np.sum(np.power(ps, 2), axis = 1)
            xc = np.dot(ps, self.centroids.T)
            z = np.sqrt(cc + xx - 2*xc)
    
            v = np.min(z, axis = 1)
            inds = np.argmin(z, axis = 1)#
            mu = np.mean(z, axis = 1)
            ps = mu - z
            ps[ps < 0] = 0
    
            off = np.asmatrix(range(0, (z.shape[0])*self.Ncentroids, self.Ncentroids))
            ps = np.zeros((ps.shape[0]*ps.shape[1],1))
            ps[off.T + inds] = 1
            ps = np.reshape(ps, (z.shape[1],z.shape[0]), 'F').T#
    
                    
            prows = self.finalDim[0] - self.rf + 1
            pcols = self.finalDim[1]- self.rf + 1
            ps = np.reshape(ps, (prows, pcols, self.Ncentroids), 'F')
                    
            XC[0, :] = FeatureDescriptor.subdivPooling(ps, subdivLevels).T
        
            XCs = np.divide(XC - self.FSMean, self.FSSd)
            return XCs
        else:
            print 'FSMean and FSSd are not available, please trained model first'
        

    
    def extractFeatures (self, X, subdivLevels = 1):
        
        print 'Extracting feature vectors using centroid PatchSet...'
        startTime = time.time()
        
        Nimages = X.shape[0]

        cc = np.asmatrix(np.sum(np.power(self.centroids,2), axis = 1).T)
        sz = self.finalDim[0] * self.finalDim[1]

        XC = np.zeros((Nimages, (4**subdivLevels)*self.Ncentroids))
        
        
        for i in range(0, Nimages):
            
            if np.mod(i, 100) == 0:
                print 'Extracting features:', i, '/', Nimages

            ps = FeatureDescriptor.im2col(np.reshape(X[i,0:sz], self.finalDim[0:2], 'F'), (self.rf, self.rf))
            ps = np.divide(ps - np.mean(ps, axis = 1), np.sqrt(np.var(ps, axis = 1) +1))
            if self.whitening:
                ps = np.dot((ps - self.M), self.P)

            xx = np.sum(np.power(ps, 2), axis = 1)
            xc = np.dot(ps, self.centroids.T)
            z = np.sqrt(cc + xx - 2*xc)

            v = np.min(z, axis = 1)
            inds = np.argmin(z, axis = 1)#
            mu = np.mean(z, axis = 1)
            ps = mu - z
            ps[ps < 0] = 0
#             print 'there'
            off = np.asmatrix(range(0, (z.shape[0])*self.Ncentroids, self.Ncentroids))
            ps = np.zeros((ps.shape[0]*ps.shape[1],1))
            ps[off.T + inds] = 1
            ps = np.reshape(ps, (z.shape[1],z.shape[0]), 'F').T#

                
            prows = self.finalDim[0] - self.rf + 1
            pcols = self.finalDim[1]- self.rf + 1
            ps = np.reshape(ps, (prows, pcols, self.Ncentroids), 'F')
                
            XC[i, :] = FeatureDescriptor.subdivPooling(ps, subdivLevels).T
            
        print 'Extracting features:', i+1, '/', Nimages    
        endTime = time.time()
        print X.shape[0], 'feature vectors computed in', endTime-startTime, 'sec\n'
        
        if self.unTrained:
            print 'Updated dictionary....'
            self.FSMean = np.mean(XC, axis = 0)
            self.FSSd = np.sqrt(np.var(XC, axis = 0) + 0.01)
            self.saveDictionaryToFile() 
        else:
            print 'Normalized by trained features'
            
        XCs = np.divide(XC - self.FSMean, self.FSSd)
            
        return XCs
    
    def saveDictionaryToFile(self):
        print 'Saving dictionary...'   
        filePath = os.path.join(self.dicPath, 'dictionaryModel')
        np.savez(filePath,
                 rfSize = self.rf,
                 finalDim = self.finalDim,
                 whitening = self.whitening,
                 Ncentroids = self.Ncentroids,
                 Mean = self.M,
                 Patch = self.P,
                 centroids = self.centroids,
                 FSMean = self.FSMean,
                 FSSd = self.FSSd
                 )
         
        print 'Dictionary Updated in', self.dicPath, '\n'
        return self.dicPath

class SVMClassifier:
    
    clssifier = None
    modelTrained = False
    modelOptimized = False
    Opt = None
    
    def __init__(self, Opt, isTrain = None, clfPath = None):
        
        self.Opt = Opt
        if isTrain is None:
            isTrain = Opt.isTrain
        
        if isTrain is False:
            if clfPath is None:
                clfPath = Opt.svmModelPath
            try:     
                clfPath = os.path.join(clfPath, 'SVMModel.pkl')
                print 'Load classifier from', clfPath
                self.classifier = joblib.load(clfPath)
                self.modelTrained = True
                
                print 'SVM Classifier loaded.'
            except:
                print 'Unable to read the trained model'
                
        else:
            if Opt.isTrain:
                print 'Untrained SVM created.'
            else:
                print 'Options is not in train mode'

    def loadSVNModel(self, modelPath):
        return 
        
    
    def saveSVMModel(self, path = None):
        
        if path is None:
            path = self.Opt.modelPath
        
        if self.modelTrained:
            print 'Saving SVM model...'
            infoFilePath = os.path.join(path, 'SVMModelInfo')
#             np.savez(infoFilePath,
#                      classNames = self.classNames,
#                      classIDs = self.classIDs
#                      )

            clfFilePath = os.path.join(path, 'SVMModel.pkl') 
            joblib.dump(self.classifier, clfFilePath)
            
            print 'SVMModel.pkl and SVMModelInfo.pkl were saved in', path
            return path
        else:
            print 'Model has not been trained first.'
    
    def predict(self, X_test):
                    
        if self.modelTrained:
            pred_label = self.classifier.predict(X_test)
            pred_probability = self.classifier.predict_proba(X_test)
            return pred_label, pred_probability
        else:
            print 'Model has not been trained first.'
        
    
    def evaluate(self, y_true, y_pred, y_proba, save = False):
        
        accuracy = metrics.accuracy_score(y_true, y_pred)
        precision = metrics.precision_score(y_true, y_true, average=None)
        recall = metrics.recall_score(y_true, y_true, average=None)
#         result = zip(self.classNames, precision, recall)
        
        print "Accruracy:", accuracy
#         print "(Class, precision, recall)", result
        print metrics.classification_report(y_true, y_pred)
        
        
    def __evaluateCVModel(self, X_train, y_train, X_test, y_test, showIterationResult = False):
        
        if self.modelOptimized:
            print("Best parameters set found on development set:")
            print self.classifier.best_estimator_, '\n'
    
            scores =  cross_validation.cross_val_score(self.classifier.best_estimator_, X_train, y_train, cv=10)
            print("10-Fold cross-validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
            print 'Holdout testing data accuracy:', self.classifier.score(X_test, y_test), '\n'
        
            if showIterationResult:
                print("Grid scores on development set:")
                for params, mean_score, scores in self.classifier.grid_scores_:
                    print("%0.3f (+/-%0.03f) for %r" % (mean_score, scores.std() / 2, params))
                    print() 
            
            print("Detailed classification report:")
            print("The model is trained on the full development set.")
            print("The scores are computed on the full evaluation set.")
            y_pred = self.classifier.predict(X_test)
            print metrics.classification_report(y_test, y_pred), '\n'
        else:
            print 'Model has not been trained first.'

    
    def trainModel(self, X, y):
        
        print 'Training Model...'
        startTime = time.time()
#         self.classNames = classNames
        # Split into training and test set (e.g., 80/20)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)
        
        # Choose estimator
#         self.estimator = svm.SVC(kernel = 'linear', probability = True)
        self.estimator = svm.SVC(probability = True)
        
        # Choose cross-validation iterator
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.25, random_state=0)
        
        
        # Tune the hyperparameters
        gammas = np.logspace(-6, -1, 10)
        self.classifier = grid_search.GridSearchCV(estimator=self.estimator, cv=cv, param_grid=dict(gamma=gammas))
        
#         tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},]
#         self.classifier = grid_search.GridSearchCV(estimator=self.estimator, cv=cv, param_grid=tuned_parameters)     
        
        
        # Train the optimized model with the split training set
        self.classifier.fit(X_train, y_train)
        self.modelOptimized = True
        
        # Evaluate 
        self.__evaluateCVModel(X_train, y_train, X_test, y_test)
         
        # Train final model with the full training set
        print 'Train final model with the full training set...'
        self.classifier.fit(X, y)
        self.modelTrained = True
        print 'Tuned Model: Full training data accuracy:', self.classifier.score(X, y)
    
        endTime = time.time()
        print 'Complete training model in ',  endTime - startTime, 'sec\n'



#########################################
class VizClassifier():
   
    Opt = None
    FeatureDescriptor = None
    Classifier = None
    
    
#     nImageAll = 0
   
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
    VCLF.classifyCouldImages()
#     VCLF.classifyLocalImagesParallel()
#     VCLF.classifyCloudImagesParallel(1000, 1100)
