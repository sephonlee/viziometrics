import os, errno
import numpy as np
import cv2 as cv
import random 
import time
import datetime
from matplotlib import pyplot as plt
from os.path import *
from sklearn.cluster import KMeans, MiniBatchKMeans
import sklearn
import json
import csv
import cPickle as pickle
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics
# from sklearn import externals
from sklearn.externals import joblib
# from sklearn.learning_curve import learning_curve


# Class of all parameters
class Opt():
    def __init__(self):
        
        
        self.optSaveDst = '/Users/sephon/Desktop/Research/VizioMetrics/DB/'
        
        
        ## Data Read Parameter
        self.finalDim = [128, 128, 1]    # Final image dimensions
        self.Ntrain = 1                  #/ Number of training images per category
        self.Ntest = 1                   #/ Number of test images per category
        self.validImageFormat = ['jpg', 'tif', 'bmp', 'png', 'tiff']    # Valid image formats
#         self.catDirs = ['photo', 'scheme', 'table', 'visualization']    # Category subdirs
        self.classNames = ['photo', 'test3','scheme', 'test2', 'table', 'visualization', 'multichart']
        self.classNames = sorted(self.classNames)
        self.classIDs = range(1, len(self.classNames)+1)        # Start from 1
        
        ## Patch Parameters
        self.Npatches = 50000;           # Number of patches
        self.Ncentroids = 200;           # Number of centroids
        self.rfSize = 6;                 # Receptor Field Size (i.e. Patch Size)
        self.kmeansIterations = 100      # Iterations for kmeans centroid computation
        self.whitening = True            # Whether to use whitening
        self.normContrast = True         # Whether to normalize patches for contrast
        self.minibatch = False           # Use minibatch to train SVM 
        self.MIN_PATCH_VAR = 38/255      # Minimum Patch Variance for accepting as potential centroid (empirically set to about 25% quartile of var)

        self.kmeansIterations = 50
        self.minibatch = False
        
        # saving path
        self.modelName = 'nClass_%d_' % len(self.classNames)
        print 'Options set!'

            
    def saveSetting(self, outPath):
        
        print 'Saving options'
        classData = {}
        for i in range(0, len(self.classNames)):
            classData[i+1] = self.classNames[i]
        
        outPath = os.path.join(outPath, 'opt.pkl')   
        with open(outPath, 'wb') as fp:
            pickle.dump(classData, fp)
        print 'Options were saved in', outPath, '\n'
        
class Common():
    
    @staticmethod
    def makeDir(dst, dirName):

        path = os.path.join(dst, dirName)        
        try:
            os.makedirs(path)
            print "Create new directory " + path
            return path    
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                print  path + ' is existed.'
                pass
            else: raise
            return path
        
    @staticmethod
    def getModelPath(path, dirName):
        dirName = dirName + datetime.datetime.now().strftime("%Y-%m-%d")
        return PatchSet.makeDir(path, dirName)
    
    
    @staticmethod
    def getFileNameAndSuffix(filePath):
        
        filename = filePath.split('/')[-1]
        suffix = filename.split('.')[1]
        
        return filename, suffix
        
    
# Class of Image Data
class DataSet():
    
    def __init__(self, opt):
        self.finalDim = opt.finalDim
#         self.Ntrain = opt.Ntrain
#         self.Ntest = opt.Ntest
        self.validImageFormat = opt.validImageFormat   
        
        self.classNames = opt.classNames
        self.classIDs = opt.classIDs
        self.classInfo = dict(zip(self.classNames, self.classIDs))
        
        
    # Load training image data from local database
    def loadTrainDataByQuerry(self):
        return
    
    # Load testing image data from local database
    def loadTestDataByQuerry(self):
        self.loadTrainDataByQuerry()
        return
    
    def LoadUnLabeledDataFromDisk(self, path):
        print "Loading Images from" + path
        
        
        # read all images
        index = 0;
        dimHeightSum = 0
        dimWidthSum = 0
        startTime =  time.time()
        
        imData, imDims, fileList, dimSum = self.__loadImages(path)
        NImages = len(imDims)
        print NImages
        

            
        return
    
    # Load testing image data from local classified directories
    # Directory name = class name
    def LoadTestDataFromCatDir(self, inPath, outPath):
        return self.loadDataFromCatDir(inPath, outPath, type = 'test')
        
        
    # Load training image data from local classified directories
    # Directory name = class name
    def loadTrainDataFromCatDir_(self, inPath, outPath):
        return self.loadDataFromCatDir(inPath, outPath, type = 'train')
        
    def loadDataFromCatDir(self, inPath, outPath, type = 'unknown'):
        print "Loading Images from" + inPath
        
        # temp variables
        index = 0;
        startTime =  time.time()
        
        # containers
        allImData = None
        catDirs = []
        catIDs = {}
        imDimsInCat = []
        fileNamesInCat = []
        numImageInCat = {}
        allLabels = []
        allCatNames = []
        
        # single parameters
        dimHeightSum = 0
        dimWidthSum = 0
        NtotalImages = 0
        maxCatSize = 0
        
        
        # get categories from directory

        for dirName in  os.listdir(inPath):
            if dirName in self.classNames:
                catDirs.append(dirName)
                catIDs[dirName] = self.classInfo[dirName]
        
        NCategories = len(catDirs)
        
        ####################
        # Loop to read files
        for catName in catDirs:
            catPath = os.path.join(inPath, catName)
            imData, imDims, fileList, dimSum = self.__loadImages(catPath)
            NImages = len(imDims)
            
            # stack all images #
            if not index:
                allImData =  imData
            else:
                allImData = np.vstack([allImData, imData])
            index += 1
            #####################
            
            # fill containers
            imDimsInCat.append(imDims)
            fileNamesInCat += fileList
            numImageInCat[catName] = NImages
            allLabels = np.hstack([allLabels, np.tile(catIDs[catName], (numImageInCat[catName]))])
            allCatNames = np.hstack([allCatNames, np.tile(catName, (numImageInCat[catName]))])
                
            # fill single parameters
            NtotalImages += NImages 
            if NImages > maxCatSize:
                maxCatSize = NImages
            dimHeightSum += dimSum[0]
            dimWidthSum += dimSum[1]
            meanDim = [dimHeightSum / float(NtotalImages), dimWidthSum / float(NtotalImages)]

            print 'Collect', NImages, 'images from', catName

        endTime = time.time()
        
        print 'Total', NtotalImages,'images collected in', endTime-startTime, 'sec'
        print 'Average image dimension: ', meanDim, '\n'
         
        # Save Info
#         dirName = 'nClass_%d_' % len(self.classNames)
#         dirName = dirName + datetime.datetime.now().strftime("%Y-%m-%d")
#         outPath = Common.makeDir(outPath, dirName) 
         
        saveContent = zip(fileNamesInCat, allCatNames, allLabels)
        csvFileName = type + '_image_files'
        self.__saveCSV(outPath, csvFileName, ['file_path', 'class_name', 'class_id'], saveContent)
    
        # output
        return allImData, allLabels, allCatNames
    
    
    
    @ staticmethod
    def __saveCSV(path, filename, header, content):
        
        print 'Saving image infomation...'
        filePath = os.path.join(path, filename) + '.csv'
        with open(filePath, 'wb') as outcsv:
            writer = csv.writer(outcsv, dialect='excel')
            writer.writerow(header)
            for c in content:
                writer.writerow(c)
                
        print filename, 'were saved in', filePath, '\n'
          
    
    # return all file paths from the given directory with given file Type
    def __loadImages(self, catPath):
        
        nx, ny, nz = self.finalDim;
        fileList = self.getFileNamesFromPath(catPath)
        imDims = [];

        # read all images and reshaping
        imData = np.zeros((len(fileList), nx*ny*nz), )

        count = 0;
        dimHeightSum = 0
        dimWidthSum = 0
        for filename in fileList:
            img = cv.imread(filename)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
            imDim = img.shape 
            dimHeightSum += imDim[0]
            dimWidthSum += imDim[1]
            
            imDims.append(imDim[:2])
            img = cv.resize(img, (self.finalDim[0], self.finalDim[1]))
            img = np.asarray(img)
#             print img.shape
            img = np.reshape(img, (1, self.finalDim[0]* self.finalDim[1]))
#             print img.shape
            imData[count, :] = img
            count += 1;
            
            dimSum = [dimHeightSum, dimWidthSum]
               
        return imData, imDims, fileList, dimSum
         
    def getFileNamesFromPath(self, path):
#         print "Get file names from ", path
        fileList = []
#         fileType = "." + fileType
        num = 0;
        for dirPath, dirNames, fileNames in os.walk(path):   
            for f in fileNames:
                extension = f.split('.')[1]
                if extension in self.validImageFormat:
                    fileList.append(os.path.join(dirPath, f))
                    num += 1
                
#         print num, "files were found"
        return fileList
        
    
    
# Class of Patch    
class PatchSet():
    
    # Parameters
    N = 50000;                  # Number of patches
    Ncentroids = 200;           # Number of centroids
    rfSize = 6;                 # Receptor Field Size (i.e. Patch Size)
    kmeansIterations = 100      # Iterations for kmeans centroid computation
    whitening = True            # Whether to use whitening
    finalDim = [121, 121, 1]    # Image Dimensions
    normContrast = True         # Whether to normalize patches for contrast
    minibatch = False           # Use batch to train SVM
    MIN_PATCH_VAR = 38/255      # Minimum Patch Variance for accepting as potential centroid (empirically set to about 25% quartile of var)
    
    # Data
    patches = None              # [N X rfSize^2] Patch Data Matrix
    patchLabels = None          # [N X 1] Vector assigning category indices to each patch
    M = None                    # Patch Mean Matrix
    P = None                    # Patch Alignment Matrix (Right-Multiplies for whitening)
    centroids = None            # Patch Centroids (computed through k-means clustering)
    centroidFrequency = None    # Centroid Occurence Frequencies
    
    def __init__(self, opt, allImData, allLabels):
        
        # Update parameters
        self.N = opt.Npatches;
        self.Ncentroids = opt.Ncentroids;
        self.rfSize = opt.rfSize;
        self.kmeansIterations = opt.kmeansIterations;
        self.whitening = opt.whitening;
        self.finalDim = opt.finalDim;
        self.minibatch = opt.minibatch
        self.classNames = opt.classNames
        self.MIN_PATCH_VAR = opt.MIN_PATCH_VAR
        
        # Data
        self.patches = None
        self.patchLabels = None
        self.M = None
        self.P = None
        self.centroids = None
        self.centroidFrequency = None
        self.extractPatch(allImData, allLabels)

    # Extract Random Patches from Data (if labels provided, save patch label as well)
    def extractPatch(self, allImData, allLabels):
        
        print 'Extracting random patches from data...'
        
        Ndata = allImData.shape[0]
        nx = self.finalDim[0]
        ny = self.finalDim[1]
        nc = self.finalDim[2]
        rf = self.rfSize;

        A = np.zeros((self.N, rf * rf * nc))
        A = np.asmatrix(A)
        L = np.ones((self.N, 1))
        L = np.asmatrix(L)
        
        startTime =  time.time()
        i = 0
        trials = 0
        maxTry = 0
        
        
        
        while i < self.N:
            if trials % 10000 == 0:
                print i, '/', self.N,  'patches accepted.'
                
            r = random.randint(0, nx - rf)
            c = random.randint(0, ny - rf)
            maxTry += 1
            index = i % Ndata # index will repeat 0, 1, 2, 3...len(imData)
            patch = np.reshape(allImData[index, :],(nx, ny, nc))
            patch = patch[r:r+rf, c:c+rf, :]
            
            if np.var(patch) > self.MIN_PATCH_VAR:
                A[i,:] = np.reshape(patch, (rf * rf * nc))
                L[i] = allLabels[index]
                i += 1
            
            trials += 1
            
        self.patches = A
        self.patchLabels = L
        endTime =  time.time()
        
        print self.N, 'patches extracted in', endTime - startTime, '\n'
        

    # K-means for Centroid Computation (uses VL_feat subroutine)
    def kmeansCentroids(self):
        normPathces, self.M, self.P = PatchSet.normalizeAndWhiten(self.patches);
        self.centroids, self.centroidFrequency = PatchSet.kmeansVL(normPathces, self.Ncentroids, self.minibatch)
        
    # show centroids
    def showCentroids(self):
        
        if self.centroids is None:
            print 'Centroids have not been computed.'
        
#         highlight = mod
        
        H = self.rfSize
        W = H
        
        NChannel = self.centroids.shape[1] / (H*W)
        
        vizMatCols = round(np.sqrt(self.Ncentroids))
        vizMatRows = np.ceil(self.Ncentroids / vizMatCols)
        
        C = PatchSet.invertWhiteningAndNormalization(self.centroids, self.M, self.P)
        
        C = (C * 40) + 190 # Approximate contrast denormalization for visibility (empirical values for mean and sqvar)
        C[C < 0] = 0
        C[C > 255] = 255
        
        if NChannel > 1:
            image = np.ones((vizMatRows*(H+1), vizMatCols*(W+1), NChannel), dtype = 'uint8')* 100
        else:
            image = np.ones((vizMatRows*(H+1), vizMatCols*(W+1)), dtype = 'uint8')* 100
        
        for i in range(self.Ncentroids):
            r = np.floor((i) / vizMatCols)
            c = i % vizMatCols
            centr = np.reshape(C[i,:], (H, W, NChannel))
            centr = centr.astype('uint8')
            
            if NChannel > 1:
                image[(r*(H+1)):((r+1)*(H+1))-1, (c*(W+1)):((c+1)*(W+1))-1, :] = centr
            else:
                image[(r*(H+1)):((r+1)*(H+1))-1, (c*(W+1)):((c+1)*(W+1))-1] = centr
        
        
        if NChannel == 3:
            b,g,r = cv.split(image)       # get b,g,r
            rgb_img = cv.merge([r,g,b]) 
            plt.imshow(rgb_img)
        else:
            plt.imshow(image, cmap = 'gray')
            
        plt.xticks([]), plt.yticks([])   # to hide tick values on X and Y axis
        plt.show()
        
        
    def saveDictionaryToFile(self, path):
        
        print 'Saving dictionary...'   
        filePath = os.path.join(path, 'dictionaryModel')
        np.savez(filePath,
                 rfSize = self.rfSize,
                 finalDim = self.finalDim,
                 Mean = self.M,
                 Patch = self.P,
                 whitening = self.whitening,
                 centroids = self.centroids,
                 Ncentroids = self.Ncentroids
                 )
         
        print 'Dictionary saved in', path, '\n'
        return path
         
        
    @staticmethod
    def makeDir(dst, dirName):

        path = os.path.join(dst, dirName)        
        try:
            os.makedirs(path)
            print "Create new directory " + path
            return path    
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                print  path + ' is existed.'
                pass
            else: raise
            return path
            
        
    @staticmethod
    # Normalize and whiten the patchset
    def normalizeAndWhiten(A):
                
        pM = np.mean(A, axis=1)
        pSqVar = np.sqrt(A.var(axis=1) + 10)
        pSqVar = pSqVar.astype('float64') 
        A = np.divide((A - pM), pSqVar)
        
        # whiten
        C = np.cov(A, rowvar = False)
        M = np.mean(A, axis = 0)
        D,V = np.linalg.eig(C)
        P = np.dot(np.dot(V, np.diag(np.sqrt(1/(D + 0.1)))), V.T)
        A = np.dot(A - M, P)
        
        return A, M, P
        
    @staticmethod
    def invertWhiteningAndNormalization(A, M, P):
        A = np.dot(A, np.linalg.inv(P)) + M
        return A
        
    @staticmethod
    def kmeansVL(X, k, minibatch):
        
        # choose k-mean algorithm
        if minibatch:
            km = MiniBatchKMeans(n_clusters = k, init='k-means++', n_init=10, init_size=1000, batch_size = 1000, verbose = 0)
        else:
#             km = KMeans(n_clusters = k, init='k-means++', max_iter = 100, n_init=10, verbose = 0)
            km = KMeans(n_clusters = k, n_init = 10, max_iter = 10, init = 'random')
        print("Clustering sparse data with %s" % km)
        
        # start training
        startTime = time.time()
        km.fit(X)
        endTime = time.time()
        
        labels = km.labels_
        centoirds = km.cluster_centers_
        
        # manipulate order
        N,bins = np.histogram(labels, np.asarray(range(0, k+1)))
        descOrderSort = np.argsort(N)[::-1][:k]
        centroids = centoirds[descOrderSort]
        cenFreq = N[descOrderSort] / float(np.sum(N))
        
        print 'Clustering completed in', endTime-startTime, 'sec\n' 
        return centroids, cenFreq
        
        
class FeatureDescriptor():
    
    
    def __init__(self, dicPath):
        
        try:
            
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

        except:
            print 'Unable to load dictionary'
        
    
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
            
    def extractFeatures (self, X, dicPath, subdivLevels = 1):
        
        print 'Extracting feature vectors using centroid PatchSet...'
        startTime = time.time()
        
        Nimages = X.shape[0]
        
        cc = np.sum(np.power(self.centroids,2), axis = 1).T  
        sz = self.finalDim[0] * self.finalDim[1]

        XC = np.zeros((Nimages, (4**subdivLevels)*self.Ncentroids))

        for i in range(0, Nimages):
            
            if np.mod(i, 1000) == 0:
                print 'Extracting features:', i, '/', Nimages
            
            for i in range(0, X.shape[0]):

                ps = FeatureDescriptor.im2col(np.reshape(X[i,0:sz], self.finalDim[0:2]), (self.rf, self.rf))  
                ps = np.divide(ps - np.mean(ps, axis = 1), np.sqrt(np.var(ps, axis = 1) +1))
#                 
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
                ps = np.reshape(ps, (z.shape[0],z.shape[1]))#
                
                prows = self.finalDim[0] - self.rf + 1
                pcols = self.finalDim[1]- self.rf + 1
                ps = np.reshape(ps, (prows, pcols, self.Ncentroids))
                
                XC[i, :] = FeatureDescriptor.subdivPooling(ps, subdivLevels).T
                
            endTime = time.time()
            print X.shape[0], 'feature vectors computed in', endTime-startTime, 'sec\n'
            
            mean = np.mean(XC, axis = 0)
            sd = np.sqrt(np.var(XC, axis = 0) + 0.01)
            XCs = np.divide(XC - mean, sd)
#             XCs = np.hstack([XCs, np.ones((XCs.shape[0],1))])
            
            return XCs
        
    @ staticmethod
    def getFeatureNames(Ncentroids):
            
        featureNames = []
        for i in range(1, Ncentroids+1):
            name = 'centroids_' + str(i)
            featureNames.append(name)
        return np.asmatrix(featureNames).T
            
class Classifier:
    
    clssifier = None
    modelTrained = False
    
    
    def __init__(self, clfPath = None):
                
        if clfPath is not None:
            try:     
                clfiPath = os.path.join(clfPath, 'SVMModelInfo.npz')
                clfPath = os.path.join(clfPath, 'SVMModel.pkl')
                print 'Load classifier from', clfPath
                clf = np.load(clfiPath)
                self.classifier = joblib.load(clfPath) 
                self.modelTrained = True
                
                print 'Classifier loaded.'
            except:
                print 'Unable to read the trained model'

        
    def saveSVMModel(self, path):
        
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
        
        
    def __evaluateCVModel(self, X_test, y_test, showIterationResult = False):
        
        if self.modelTrained:
            print("Best parameters set found on development set:")
            print(self.classifier.best_estimator_)
            print 'Cross validation accuracy:', self.classifier.score(X_test, y_test)
        
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
        self.estimator = svm.SVC(kernel = 'linear', probability = True)
        
        # Choose cross-validation iterator
        cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.25, random_state=0)
        
        # Tune the hyperparameters
        gammas = np.logspace(-6, -1, 10)
        
        tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},]
        
#         self.classifier = grid_search.GridSearchCV(estimator=self.estimator, cv=cv, param_grid=tuned_parameters)     
        self.classifier = grid_search.GridSearchCV(estimator=self.estimator, cv=cv, param_grid=dict(gamma=gammas))
        
        # Train the model with optimized params by the training set
        self.classifier.fit(X_train, y_train)
        self.modelTrained = True
        # Evaluate 
        self.__evaluateCVModel(X_test, y_test)
         
         
        # Train final model on the full dataset for future use
        self.classifier.fit(X, y)
        print 'Tuned Model: Full training data accuracy:', self.classifier.score(X, y)
    
        endTime = time.time()
        print 'Complete training model in ',  endTime - startTime
          
        
        
if __name__ == '__main__':   

        
    
    corpusPath = "/Users/sephon/Desktop/Research/ReVision/corpus/VizSet_pm_ee_cat014_sub"
    workshopPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model"
# 
    opt = Opt()
    modelPath = Common.getModelPath(workshopPath, opt.modelName)
    opt.saveSetting(modelPath)
    
    DS = DataSet(opt)
    allImData, allLabels, allCatNames = DS.loadTrainDataFromCatDir_(corpusPath, modelPath)

    ps = PatchSet(opt, allImData, allLabels)
    ps.kmeansCentroids()
#     ps.showCentroids()
   
    dicPath = ps.saveDictionaryToFile(modelPath)
      
      
      
  
    FD = FeatureDescriptor(dicPath)
    X = FD.extractFeatures(allImData, 1)
    y = allCatNames
       
    SVM = Classifier()
    SVM.trainModel(X, y)
    y_pred, y_proba = SVM.predict(X)
    SVM.evaluate(y, y_pred, y_proba)
   
   
     
    modelPath = SVM.saveSVMModel(modelPath)

#   
    modelPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model/nClass_7_2014-10-11'
#     print modelPath
    SVM_ = Classifier(clfPath = modelPath)
      
    y_pred, y_proba = SVM_.predict(X)
    SVM_.evaluate(y, y_pred, y_proba)
      
    
    
    
    
    
    
    
    
    
    
    
    
#     #     print allCatNames
# #     featureName = FeatureDescriptor.getFeatureNames(allImData.shape[1])
# 
#     # Split into training and test set (e.g., 80/20)
#     X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
#     
#     # Choose estimator
#     estimator = svm.SVC(kernel = 'rbf')
#     
#     # Choose cross-validation iterator
#     cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
#     
#     # Tune the hyperparameters
#     gammas = np.logspace(-6, -1, 10)
#     classifier = grid_search.GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
#     
#     classifier.fit(X_train, y_train)
#      
# #     # Debug algorithm with learning curve
# #     title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
# #     estimator = svm.SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
# # #     learning_curve.plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
# #     plt.show()
#      
#     # Final evaluation on the test set
#     print classifier.score(X_test, y_test)
#     print 
#      
#     #  Test over-fitting in model selection with nested cross-validation (using the whole dataset)
# #     cross_validation.cross_val_score(classifier, X, Y)
#      
#     # Train final model on whole dataset
#     classifier.fit(X, Y)
#     print classifier.score(X, Y)
# 
#     estimator.fit(X, Y)
#     print estimator.score(X, Y)
# 
# #     print classifier.decision_function(X)
# 
#     result =  classifier.predict(X_test)
#     print result.shape
#     print result
#     fileList = DataSet.getFileNamesFromPath(path)
#     print fileList
    
    
    
            # Debug algorithm with learning curve
#         title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
#         estimator = svm.SVC(kernel='rb', gamma=classifier.best_estimator_.gamma)
#         learning_curve.plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
#         plt.show()
         
