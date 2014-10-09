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
import cPickle as pickle
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
# from sklearn.learning_curve import learning_curve


# Class of all parameters
class Opt():
    def __init__(self):
        
        self.finalDim = [128, 128, 1]    # Final image dimensions
        self.Ntrain = 1                  #/ Number of training images per category
        self.Ntest = 1                   #/ Number of test images per category
        self.validImageFormat = ['jpg', 'tif', 'bmp', 'png', 'tiff']    # Valid image formats
        self.catDirs = ('photo', 'scheme', 'table', 'visualization')    # Category subdirs


        self.Npatches = 50000;                  # Number of patches
        self.Ncentroids = 200;           # Number of centroids
        self.rfSize = 6;                 # Receptor Field Size (i.e. Patch Size)
        self.kmeansIterations = 100      # Iterations for kmeans centroid computation
        self.whitening = True            # Whether to use whitening
        self.normContrast = True         # Whether to normalize patches for contrast
        self.minibatch = False           # Use minibatch to train SVM 
        self.MIN_PATCH_VAR = 38/255      # Minimum Patch Variance for accepting as potential centroid (empirically set to about 25% quartile of var)

        self.kmeansIterations = 50
        self.minibatch = False
        
        print 'Options set!'
       
        
        
# Class of Image Data
class DataSet():
    
    
    def __init__(self, opt):
        self.finalDim = opt.finalDim
        self.Ntrain = opt.Ntrain
        self.Ntest = opt.Ntest
        self.validImageFormat = opt.validImageFormat
        self.catDirs = opt.catDirs
        self.NCategories = len(self.catDirs)
        self.labels = range(1, self.NCategories + 1) # Ground truth label
        
        # set by loadData
        self.NtotalImages = 0  # Total number of images
        self.maxCatSize = 0 # Maximum number of images among categories 
        self.meanDim = [] # Average dimension
        self.imDimsInCat = () # Image dimensions in each category
        self.fileNamesInCat = () # Filenames in each category
        self.numImageInCat = () # Total number of images in each category
        self.allImData = np.array([]) # All image data
        self.dataLoaded = False; 
        
        print 'DataSet Created'
        
    def loadData(self, path):
        print "Loading Images from" + path
        
        
        # read all images
        index = 0;
        dimHeightSum = 0
        dimWidthSum = 0
        startTime =  time.time()
        
        for cat in self.catDirs:
            catPath = os.path.join(path,cat)
            imData, imDims, fileList, dimSum = self.__loadImages(catPath)
            NImages = len(imDims)
            
            # stack all images #
            if not index:
                self.allImData =  imData
            else:
                self.allImData = np.vstack([self.allImData, imData])
            index += 1
            #####################
            
            self.imDimsInCat += (imDims,)
            self.fileNamesInCat += (fileList,)
            self.numImageInCat += (NImages,)
            self.NtotalImages += NImages 

            if NImages > self.maxCatSize:
                self.maxCatSize = NImages

            dimHeightSum += dimSum[0]
            dimWidthSum += dimSum[1]
            self.meanDim = [dimHeightSum / float(self.NtotalImages), dimWidthSum / float(self.NtotalImages)]

               
        self.dataLoaded = True
        endTime = time.time()
        for i in range(0, self.NCategories):
            print 'Collect', self.numImageInCat[i], 'images from', self.catDirs[i]
        
        print 'Total', self.NtotalImages,'images collected in', endTime-startTime, 'sec'
        print 'Average image dimension: ', self.meanDim, '\n'
        

          
            
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
        
    def getCatName(self, i):
        return self.catDirs[i]
        
    def getWSFull(self):
        
        for i in range(0, self.NCategories):
            if not i:
                allLabels = np.tile(self.labels[i], (self.numImageInCat[i]))
#                 allLabels = np.tile(self.labels[i], (self.numImageInCat[i], 1))
                allCatNames = np.tile(self.catDirs[i], (self.numImageInCat[i], 1))
            else:
#                 allLabels = np.vstack([allLabels, np.tile(self.labels[i], (self.numImageInCat[i], 1))])
                allLabels = np.hstack([allLabels, np.tile(self.labels[i], (self.numImageInCat[i]))])
                allCatNames = np.vstack([allCatNames, np.tile(self.catDirs[i], (self.numImageInCat[i], 1))])
                        
        return self.allImData, allLabels, allCatNames
    
    
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
        self.MIN_PATCH_VAR = opt.MIN_PATCH_VAR
        
        # Data
        self.patches = None
        self.patchLabels = None
        self.M = None
        self.P = None
        self.centroids = None
        self.centroidFrequency = None
        
        self.extractPatch(allImData, allLabels)
        
    
#         N = 50000;              % Number of patches
#         Ncentroids = 150;       % Number of centroids
#         rfSize = 6;             % Receptor Field Size (i.e. Patch Size)
#         whitening = true;       % Whether to use whitening
#         normContrast = true;    % Whether to normalize patches for contrast
#         DIM = [128 128 1]       % Image Dimensions
#         kmeansIterations = 100  % Iterations for kmeans centroid computation
#         patches = [];           % [N X rfSize^2] Patch Data Matrix
#         patchLabels = [];       % [N X 1] Vector assigning category indices to each patch
#         M = [];                 % Patch Mean Matrix
#         P = [];                 % Patch Alignment Matrix (Right-Multiplies for whitening)
#         centroids = [];         % Patch Centroids (computed through k-means clustering)
#         centroidFrequency = []; % Centroid Occurence Frequencies

    # Extract Random Patches from Data (if labels provided, save patch label as well)
    def extractPatch(self, allImData, allLabels):
        
        print 'Extracting random patches from data...\n'
        
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
        
        print 'Save model in ' + path
        # make dir in model directory
        dirName = 'rf_%d_' % self.rfSize
        dirName = dirName + datetime.datetime.now().strftime("%Y-%m-%d")
        path = PatchSet.makeDir(path, dirName)
        
        path = path + 'dictionaryModel'
        np.savez(path,
                 rfSize = self.rfSize,
                 finalDim = self.finalDim,
                 Mean = self.M,
                 Patch = self.P,
                 whitening = self.whitening,
                 centroids = self.centroids,
                 Ncentroids = self.Ncentroids
                 )
         
        print 'Model saved.\n'
         
        
    @staticmethod
    def makeDir(dst, dirName):
        
        if dst[-1] in ('/', '\\'):
                path = dst + dirName + '/'
        else:
                path = dst + '/' + dirName + '/'
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
    
    
    def __init__(dicPath):
        return
    
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
            
    @ staticmethod
    def extractFeatures (X, dicPath, subdivLevels = 1):
        
        print 'Extracting feature vectors using centroid PatchSet...'
        startTime = time.time()
        
#         if dicPath[-1] in ('/', '\\'):
#                 dicPath = dicPath + 'dictionaryModel.npz'
#         else:
#                 dicPath = dicPath + '/' + 'dictionaryModel.npz'
#         
#         print dicPath
#         dictionary = np.load(dicPath)
#         
#         rf = dictionary['rfSize']
#         finalDim = dictionary['finalDim']
#         M = dictionary['Mean']
#         P = dictionary['Patch']
#         whitening = dictionary['whitening']
#         centroids = dictionary['centroids']
#         Ncentroids = dictionary['Ncentroids']
#         Nimages = X.shape[0]
        
        cc = np.sum(np.power(centroids,2), axis = 1).T  
        sz = finalDim[0] * finalDim[1]
        XC = np.zeros((Nimages, (4**subdivLevels)*Ncentroids))

        for i in range(0, Nimages):
            
            if np.mod(i, 1000) == 0:
                print 'Extracting features:', i, '/', Nimages, '\n'
            
            for i in range(0, X.shape[0]):

                ps = FeatureDescriptor.im2col(np.reshape(X[i,0:sz], finalDim[0:2]), (rf, rf))  
                ps = np.divide(ps - np.mean(ps, axis = 1), np.sqrt(np.var(ps, axis = 1) +1))
#                 
                if whitening:
                    ps = np.dot((ps - M),P)
                    
                xx = np.sum(np.power(ps, 2), axis = 1)
                xc = np.dot(ps, centroids.T)
                z = np.sqrt(cc + xx - 2*xc)
                
                v = np.min(z, axis = 1)
                inds = np.argmin(z, axis = 1)#
                mu = np.mean(z, axis = 1)
                ps = mu - z
                ps[ps < 0] = 0
                
                off = np.asmatrix(range(0, (z.shape[0])*Ncentroids, Ncentroids))
                ps = np.zeros((ps.shape[0]*ps.shape[1],1))
                ps[off.T + inds] = 1
                ps = np.reshape(ps, (z.shape[0],z.shape[1]))#
                
                prows = finalDim[0] - rf + 1
                pcols = finalDim[1]- rf + 1
                ps = np.reshape(ps, (prows, pcols, Ncentroids))
                
                XC[i, :] = FeatureDescriptor.subdivPooling(ps, subdivLevels).T
                
#                 if int(i/float(Nimages)*100) in range(0,110, 10)[1:]:
#                     print i/Nimages, '% images were featured...'
                
            endTime = time.time()

            print X.shape[0], 'feature vectrs computed in', endTime-startTime, 'sec\n'
            
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
    
            
        
        
if __name__ == '__main__':   


    finalDim = [121, 121, 1];
    path = "/Users/sephon/Desktop/Research/ReVision/corpus/VizSet_pm_ee_cat014_sub/"
#     opt = {'finalDim': [121, 121, 1],'Ntrain': 1, 'Ntest':1,  'validImageFormat': ['jpg', 'tif', 'bmp', 'png', 'tiff'], 'catDirs': ('photo', 'scheme', 'table', 'visualization')}
 
    opt = Opt();
    a = DataSet(opt);
    a.loadData(path);
    allImData, allLabels, allCatNames = a.getWSFull();
    print np.asarray(allLabels).shape
       
    ps = PatchSet(opt, allImData, allLabels)
    ps.kmeansCentroids()
#     ps.showCentroids()
 
    dicPath = "/Users/sephon/Desktop/Research/VizioMetrics/Dictionary/"
    ps.saveDictionaryToFile(dicPath)
    dicPath = "/Users/sephon/Desktop/Research/VizioMetrics/Dictionary/rf_6_2014-10-05"
     
    X = FeatureDescriptor.extractFeatures(allImData, dicPath, 1)
    Y = allLabels
    
    print allLabels.shape
#     print allCatNames
#     featureName = FeatureDescriptor.getFeatureNames(allImData.shape[1])

    # Split into training and test set (e.g., 80/20)
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, Y, test_size=0.2, random_state=0)
    
    # Choose estimator
    estimator = svm.SVC(kernel = 'linear')
    
    # Choose cross-validation iterator
    cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.2, random_state=0)
    
    # Tune the hyperparameters
    gammas = np.logspace(-6, -1, 10)
    classifier = grid_search.GridSearchCV(estimator=estimator, cv=cv, param_grid=dict(gamma=gammas))
    
    classifier.fit(X_train, y_train)
     
#     # Debug algorithm with learning curve
#     title = 'Learning Curves (SVM, linear kernel, $\gamma=%.6f$)' %classifier.best_estimator_.gamma
#     estimator = svm.SVC(kernel='linear', gamma=classifier.best_estimator_.gamma)
# #     learning_curve.plot_learning_curve(estimator, title, X_train, y_train, cv=cv)
#     plt.show()
     
    # Final evaluation on the test set
    print classifier.score(X_test, y_test)
     
    #  Test over-fitting in model selection with nested cross-validation (using the whole dataset)
#     cross_validation.cross_val_score(classifier, X, Y)
     
    # Train final model on whole dataset
    classifier.fit(X, Y)
    print classifier.score(X, Y)

    estimator.fit(X, Y)
    print estimator.score(X, Y)

#     fileList = DataSet.getFileNamesFromPath(path)
#     print fileList
    