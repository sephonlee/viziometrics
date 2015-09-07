import sys
sys.path.append("..")
from Classifier.Options import Common
import os

# Class of all parameters
class Option_Dismantler():
    def __init__(self, isTrain = False, isTest = False, isClassify = False):
        
        ## Mode
        logic = isTrain + isTest + isClassify
        if logic == 0:
            isClassify = True # Default
        elif logic > 1:
            print 'Warning! Must select either training mode or classifying mode'
        
        if isTrain:
            print 'Training Mode'
        elif isTest:
            print 'Testing Mode'
        elif isClassify:
            print 'Classifying Mode'
            
        self.isTrain = isTrain
        self.isTest = isTest
        self.isClassify = isClassify
        self.validImageFormat = ['jpg', 'tif', 'bmp', 'png', 'tiff']
        
        #Threshold
        self.thresholds = {'splitThres': 0.999, 'varThres': 3, 'var2Thres': 100}

        ## Dismantler
        ##### Split #####
        self.isPreClassified = False
        
        ## SVM Classifier Parameter
        ##### Training #####
#         self.tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0, 1e-3, 1e-4], 'C': [1, 10]},]
        self.tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0, 1e-3, 1e-4], 'C': [1]},]
#         self.tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},]
        self.classNames = ['standalone', 'auxiliary']
        
        
        if self.isTrain:
            ## New Model Name
            self.modelName = 'dismantler_matsplit_matsvm_ceil'
            ## Corpus Path
#             self.trainCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/Corpus/train_corpus"
            self.trainCorpusPath = '/home/ec2-user/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat1_multi_subimages'
            ## Model Saving Path
#             self.modelSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model/Dismantler"
            self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model/Dismantler'
            ## New Model Path
            self.modelPath = Common.getModelPath(self.modelSavingPath, self.modelName)   
        
        ##### Classifying #####
        if self.isClassify:
            ## Model ID
            self.modelName = 'dismantler_matsplit_matsvm_ceil_latest'
            ## Model Saving Path
#             self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model/Dismantler'
            self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model/Dismantler'
            ## Default Dictionary Path
            self.dicPath = os.path.join(self.modelSavingPath, self.modelName)
            ## Default SVM Model Path
            self.svmModelPath = os.path.join(self.modelSavingPath, self.modelName)
            
            
        print 'Options set!\n'
        
        
# Class of all parameters
class Option_CompositeDetector():
    def __init__(self, isTrain = False, isTest = False, isClassify = False):
        
        ## Mode
        logic = isTrain + isTest + isClassify
        if logic == 0:
            isClassify = True # Default
        elif logic > 1:
            print 'Warning! Must select either training mode or classifying mode'
        
        if isTrain:
            print 'Training Mode'
        elif isTest:
            print 'Testing Mode'
        elif isClassify:
            print 'Classifying Mode'
            
        self.isTrain = isTrain
        self.isTest = isTest
        self.isClassify = isClassify
        self.validImageFormat = ['jpg', 'tif', 'bmp', 'png', 'tiff']
        self.classNames = ['single', 'composite']
        self.classNames = sorted(self.classNames)
        self.classIDs = range(1, len(self.classNames)+1)        # Start from 1
        self.classInfo = dict(zip(self.classNames, self.classIDs))
        
        if self.isTrain or self.isClassify:
            print 'Warning! Must select either training mode or classifying mode'
        
        ## Descriptor Parameter
        # Size Normalization
        self.offset_dim = (512, 581)
        
        # Firelane approach
        self.num_cut = 12
        self.thresholds = {'splitThres': 0.999, 'varThres': 3, 'var2Thres': 100}
        
        # Firelane map approach
        self.division = (10, 10)

        ## Classifiers
        self.availableClassifiers = ['SVM', 'CNN']
        self.activatedClassifiers = 'SVM'
        self.tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [0, 1e-3, 1e-4], 'C': [1, 10, 100, 1000]},]        
        
        ##### Training #####
        if self.isTrain:
            ## New Model Name
            self.modelName = 'nClass_%d_' % len(self.classNames)
            ## Corpus Path
#             self.trainCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/ee_cat1_multi_subimages"
            self.trainCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/Dismantler/ee_cat1_multi_subimages"
            ## Model Saving Path
#             self.modelSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model/Dismantler"
            self.modelSavingPath = "/home/ec2-user/VizioMetrics/Model/Dismantler"
            ## New Model Path
            self.modelPath = Common.getModelPath(self.modelSavingPath, self.modelName)
        
        
        ##### Classifying #####
        if self.isClassify:
            ## Model ID
#             self.modelName = 'nClass_6_2014-10-30'
            self.modelName = 'composite_detector_firelanemap'
            ## Model Saving Path
#             self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model/Dismantler'
            self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model/Classifier'
            ## Default SVM Model Path
            self.modelPath = os.path.join(self.modelSavingPath, self.modelName)
            
        print 'Options set!\n'
