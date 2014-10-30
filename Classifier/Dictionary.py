import numpy as np
import cv2 as cv
import time
import random 
import os, errno
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
import multiprocessing as mp

class DictionaryExtractor:
    
    # Class of Patch    

    Opt = None
    
    
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
    
    def __init__(self, Opt):
        
        self.Opt = Opt
        
        # Update parameters
        self.N = Opt.Npatches;
        self.Opt.Ncentroids = Opt.Ncentroids;
        self.Opt.rfSize = Opt.rfSize;
        self.Opt.kmeansIterations = Opt.kmeansIterations;
        self.Opt.whitening = Opt.whitening;
        self.Opt.finalDim = Opt.finalDim;
        self.Opt.minibatch = Opt.minibatch
        self.Opt.classNames = Opt.classNames
        self.Opt.MIN_PATCH_VAR = Opt.MIN_PATCH_VAR
        
        # Data
        self.patches = None
        self.patchLabels = None
        self.M = None
        self.P = None
        self.centroids = None
        self.centroidFrequency = None

    def getLocalDictionaryPath(self, allImData, allLabels, outPath = None):
        
        if outPath is None:
            outPath = self.Opt.modelPath
        # Processing
        self.extractPatch(allImData, allLabels)
        self.kmeansCentroids()
        return self.saveDictionaryToFile(outPath)
        
    # Extract Random Patches from Data (if labels provided, save patch label as well)
    def extractPatch(self, allImData, allLabels):
        
        print 'Extracting random patches from data...'
        
        Ndata = allImData.shape[0]
        nx, ny, nc = self.Opt.finalDim
        rf = self.Opt.rfSize;

        A = np.zeros((self.Opt.Npatches, rf * rf * nc))
        A = np.asmatrix(A)
        L = np.ones((self.Opt.Npatches, 1))
        L = np.asmatrix(L)
        
        startTime =  time.time()
        i = 0
        trials = 0
        maxTry = 0
        
        while i < self.Opt.Npatches:
            if trials % 10000 == 0:
                print i, '/', self.N,  'patches accepted.'
                
            r = random.randint(0, nx - rf)
            c = random.randint(0, ny - rf)
            maxTry += 1
            index = i % Ndata # index will repeat 0, 1, 2, 3...len(imData)
            patch = np.reshape(allImData[index, :],(nx, ny, nc), 'F')
            patch = patch[r:r+rf, c:c+rf, :]
            
            if np.var(patch) > self.Opt.MIN_PATCH_VAR:
                A[i,:] = np.reshape(patch, (rf * rf * nc), 'F')
                L[i] = allLabels[index]
                i += 1
            
            trials += 1
            
        self.patches = A
        self.patchLabels = L
        endTime =  time.time()
        
        print self.Opt.Npatches, 'patches extracted in', endTime - startTime, 'sec\n'
        
    # K-means for Centroid Computation (uses VL_feat subroutine)
    def kmeansCentroids(self):
        normPathces, self.M, self.P = DictionaryExtractor.normalizeAndWhiten(self.patches);
        self.centroids, self.centroidFrequency = DictionaryExtractor.kmeansVL(normPathces, self.Opt.Ncentroids, self.Opt.minibatch)
        
    # show centroids
    def showCentroids(self):
        
        if self.centroids is None:
            print 'Centroids have not been computed.'
        
#         highlight = mod
        
        H = self.Opt.rfSize
        W = H
        
        NChannel = self.centroids.shape[1] / (H*W)
        
        vizMatCols = round(np.sqrt(self.Opt.Ncentroids))
        vizMatRows = np.ceil(self.Opt.Ncentroids / vizMatCols)
        
        C = DictionaryExtractor.invertWhiteningAndNormalization(self.centroids, self.M, self.P)
        
        C = (C * 40) + 190 # Approximate contrast denormalization for visibility (empirical values for mean and sqvar)
        C[C < 0] = 0
        C[C > 255] = 255
        
        if NChannel > 1:
            image = np.ones((vizMatRows*(H+1), vizMatCols*(W+1), NChannel), dtype = 'uint8')* 100
        else:
            image = np.ones((vizMatRows*(H+1), vizMatCols*(W+1)), dtype = 'uint8')* 100
        
        for i in range(self.Opt.Ncentroids):
            r = np.floor((i) / vizMatCols)
            c = i % vizMatCols
            centr = np.reshape(C[i,:], (H, W, NChannel), 'F')
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
                 rfSize = self.Opt.rfSize,
                 finalDim = self.Opt.finalDim,
                 whitening = self.Opt.whitening,
                 Ncentroids = self.Opt.Ncentroids,
                 Mean = self.M,
                 Patch = self.P,
                 centroids = self.centroids,
                 FSMean = 'unavailable',
                 FSSd = 'unavailable'
                 )
         
        print 'Dictionary saved in', path, '\n'
        return path
         
        
    @staticmethod
    # Normalize and whiten the DictionaryExtracter
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
            km = MiniBatchKMeans(n_clusters = k, n_init=10, batch_size = 1000, init = 'random')
        else:
#             km = KMeans(n_clusters = k, init='k-means++', max_iter = 100, n_init=10, verbose = 0)
            km = KMeans(n_clusters = k, n_init = 10, max_iter = 50, init = 'random')
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
        