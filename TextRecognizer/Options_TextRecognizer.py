# All parameter setting
# All commonly-used functions about files
import sys
sys.path.append("..")

from Classifier.Options import Common
import cPickle as pickle
import os, errno


# Class of all parameters
class Option_TextRecognizer():
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
        
        if self.isTrain or self.isClassify:
            print 'Warning! Must select either training mode or classifying mode'
        
        ## Classifiers
        self.availableClassifiers = ['SVM', 'CNN']
        self.activatedClassifiers = 'SVM'

        ## S3 Data Read Parameter
        self.keyPath = '/Users/sephon/Desktop/Research/VizioMetrics/keys.txt'
#         self.keyPath = '/home/ec2-user/VizioMetrics/keys.txt'
        self.host = 'escience.washington.edu.viziometrics'

        ## Data Read Parameter
        self.finalDim = [32, 32, 1]    # Final image dimensions
#         self.Ntrain = 1                  #/ Number of training images per category
#         self.Ntest = 1                   #/ Number of test images per category
        self.validImageFormat = ['jpg', 'tif', 'bmp', 'png', 'tiff']    # Valid image formats
        self.classNames = self.getTextDirNames(1,40)
#         self.classNames = ['bar', 'boxplot', 'heatmap', 'line', 'pie', 'scatter']
        self.classNames = sorted(self.classNames)
        self.classIDs = range(1, len(self.classNames)+1)        # Start from 1
        self.classInfo = dict(zip(self.classNames, self.classIDs))
        
        
        ## Dictionary Parameter
        self.Npatches = 200000;             # Number of patches
        self.Ncentroids = 40;              # Number of centroids
        self.rfSize = 8;                    # Receptor Field Size (i.e. Patch Size)
        self.kmeansIterations = 100         # Iterations for kmeans centroid computation
        self.whitening = True               # Whether to use whitening
        self.normContrast = True            # Whether to normalize patches for contrast
        self.minibatch = True              # Use minibatch to train SVM 
        self.MIN_PATCH_VAR = float(38)/255  # Minimum Patch Variance for accepting as potential centroid (empirically set to about 25% quartile of var)
        self.MAX_TRY = 30                   # Maximum number of try to find a qualify patch from an image
        self.kmeansIterations = 50
        
        ## SVM Classifier Parameter
        
        
        ##### Training #####
        if self.isTrain:
            ## New Model Name
            self.modelName = 'nClass_%d_' % len(self.classNames)
            ## Corpus Path
            self.trainCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/TextRecognizer/English/Fnt"
#             self.trainCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/TextRecognizer/English/Fnt"
            ## Model Saving Path
            self.modelSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model/TextRecognizer"
#             self.modelSavingPath = "/home/ec2-user/VizioMetrics/Model/TextRecognizer"
            ## New Model Path
            self.modelPath = Common.getModelPath(self.modelSavingPath, self.modelName)
        
        ##### Testing ######
        if self.isTest:
            ## Model ID
            self.modelName = 'nClass_7_2014-10-19_4cat'
            ## Model Saving Path
            self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model'
#             self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model'
            ## Corpus Path
            self.testCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/VizSet_pm_ee_cat014_test"
#             self.testCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/VizSet_pm_ee_cat014_test"
            ## Result Directory
            self.resultSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/class_result'
#             self.resultSavingPath = '/home/ec2-user/VizioMetrics/class_result'
            ## Default Dictionary Path
            self.dicPath = os.path.join(self.modelSavingPath, self.modelName)
            ## Default SVM Model Path
            self.svmModelPath = os.path.join(self.modelSavingPath, self.modelName)
            ## Assign new folder as result directory
            self.resultPath  = Common.getModelPath(self.resultSavingPath, '')

        
        
        ##### Classifying #####
        if self.isClassify:
            ## Model ID
#             self.modelName = 'nClass_6_2014-10-30'
            self.modelName = 'nClass_7_2014-10-19_4cat'
            ## Model Saving Path
            self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model/TextRecognizer'
#             self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model'
            ## Corpus Path
            self.classifyCorpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/sketchCorpus'
#             self.classifyCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/VizSet_pm_ee_cat014_test"
            ## Result Directory
            self.resultSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/class_result'
#             self.resultSavingPath = '/home/ec2-user/VizioMetrics/class_result'
            ## Default Dictionary Path
            self.dicPath = os.path.join(self.modelSavingPath, self.modelName)
            ## Default SVM Model Path
            self.svmModelPath = os.path.join(self.modelSavingPath, self.modelName)
            ## Assign new folder as result directory
            self.resultPath  = Common.getModelPath(self.resultSavingPath, '')
            
        print 'Options set!\n'

    @ staticmethod
    def getTextDirNames(start, end):
        prefix = 'Sample'
        dirNames = []
        for i in range(start, end):
            digitLength = len(str(i))
            zerosLength = 3 - digitLength
            dirName =  prefix + '0'*zerosLength + str(i)
            dirNames.append(dirName)
        return dirNames

    def updateClassNames(self, classNames):
        self.classNames = classNames
#         self.classNames = ['bar', 'boxplot', 'heatmap', 'line', 'pie', 'scatter']
        self.classNames = sorted(self.classNames)
        self.classIDs = range(1, len(self.classNames)+1)        # Start from 1
        self.classInfo = dict(zip(self.classNames, self.classIDs))
        
    def saveSetting(self, outPath = None):
        if outPath is None:
            outPath = self.modelPath
        
        print 'Saving options'
        classData = {}
        for i in range(0, len(self.classNames)):
            classData[i+1] = self.classNames[i]
        
        outPath = os.path.join(outPath, 'opt.pkl')   
        with open(outPath, 'wb') as fp:
            pickle.dump(classData, fp)
        print 'Options were saved in', outPath, '\n'