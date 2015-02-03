from Classifier.Options import Common
import os, errno

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
        self.preClassified = False
        
        ## SVM Classifier Parameter
        ##### Training #####
        
        self.classNames = ['standalone', 'auxiliary']
        
        
        if self.isTrain:
            ## New Model Name
            self.modelName = 'dismantler_'
            ## Corpus Path
#             self.trainCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/VizSet_pm_ee_cat014"
            self.trainCorpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/train_corpus'
            ## Model Saving Path
#             self.modelSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model"
            self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/Model'
            ## New Model Path
            self.modelPath = Common.getModelPath(self.modelSavingPath, self.modelName)
        
        ##### Testing ######
        if self.isTest:
            ## Model ID
            self.modelName = 'nClass_7_2014-10-19_4cat'
            ## Model Saving Path
            self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/Model'
#             self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model'
            ## Corpus Path
            self.testCorpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/VizSet_pm_ee_cat014_test'
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
            self.modelName = 'dismantler_2014-12-18'
            ## Model Saving Path
            self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/Model'
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