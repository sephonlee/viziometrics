import os, errno
import numpy as np
import cv2 as cv
import random 
import time
import datetime
from Options import *
from DataManager import *
from matplotlib import pyplot as plt
from os.path import *
from sklearn.cluster import KMeans, MiniBatchKMeans
import sklearn
import json
import csv
from Dictionary import *
import cPickle as pickle
from sklearn import svm
from sklearn import cross_validation
from sklearn import grid_search
from sklearn import metrics
# from sklearn import externals
from sklearn.externals import joblib
# from sklearn.learning_curve import learning_curve

    

        
        
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
            
            if np.mod(i, 100) == 0:
                print 'Extracting features:', i, '/', Nimages
            
#             for i in range(0, X.shape[0]):

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
#       XCs = np.hstack([XCs, np.ones((XCs.shape[0],1))])
            
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
    Opt = None
    
    def __init__(self, Opt, clfPath = None):
                
        
        if clfPath is not None:
            try:     
#                 clfiPath = os.path.join(clfPath, 'SVMModelInfo.npz')
                clfPath = os.path.join(clfPath, 'SVMModel.pkl')
                print 'Load classifier from', clfPath
#                 clf = np.load(clfiPath)
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

        
    
#     corpusPath = "/Users/sephon/Desktop/Research/ReVision/corpus/VizSet_pm_ee_cat014_sub"
#     workshopPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model"
#
    Opt = Opt()
    modelPath = Opt.modelPath
    Opt.saveSetting(modelPath)
    
#     DS = DataSet(opt)
#     allImData, allLabels, allCatNames = DS.loadTrainDataFromCatDir_(opt.trainCorpusPath, modelPath)

    ImgLoader = ImageLoader(Opt)
    allImData, allLabels, allCatNames = ImgLoader.loadTrainDataFromLocalClassDir(Opt.trainCorpusPath, modelPath)

    Dictionary = DictionaryExtractor(Opt)
    dicPath = Dictionary.getLocalDictionaryPath(allImData, allLabels, modelPath)
    
    

#     ps = PatchSet(Opt, allImData, allLabels)
#     ps.kmeansCentroids()
#     ps.showCentroids()
   
#     dicPath = ps.saveDictionaryToFile(modelPath)
    print dicPath
      
      
  
    FD = FeatureDescriptor(dicPath)
    X = FD.extractFeatures(allImData, 1)
#     y = allCatNames
    y = allLabels
       
    SVM = Classifier()
    SVM.trainModel(X, y)
    y_pred, y_proba = SVM.predict(X)
    SVM.evaluate(y, y_pred, y_proba)
   
   
     
    modelPath = SVM.saveSVMModel(modelPath)

#   
#     modelPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model/nClass_7_2014-10-11'
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
         
