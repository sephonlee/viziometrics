# All parameter setting
# All commonly-used functions about files
import cPickle as pickle
import os, errno


# Class of all parameters
class Setting():
    def __init__(self):
        
        ## Classifiers
#         self.activatedClassifiers = 'SVM'
#         self.tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},]

        ## S3 Data Read Parameter
        self.keyPath = '/Users/sephon/Desktop/Research/VizioMetrics/keys.txt'
#         self.keyPath = '/home/ec2-user/VizioMetrics/keys.txt'
        self.host = 'escience.washington.edu.viziometrics'

        ## DB Info
        self.DBInfoPath = '/Users/sephon/Desktop/Research/VizioMetrics/db_info.txt'
#         self.DBInfoPath = '/home/ec2-user/VizioMetrics/db_info.txt'

        ## Data Read Parameter
        self.finalDim = [128, 128, 1]    # Final image dimensions
        self.preserveAspectRatio = True
        self.validImageFormat = ['jpg', 'tif', 'bmp', 'png', 'tiff']    # Valid image formats
        self.validMinKeySize = 1000000000
#         self.classNames = ['equation', 'photo','scheme', 'table', 'visualization']
# #         self.classNames = ['bar', 'boxplot', 'heatmap', 'line', 'pie', 'scatter']
# #         self.classNames = ['composite', 'single']
#         self.classNames = sorted(self.classNames)
#         self.classIDs = range(1, len(self.classNames)+1)        # Start from 1
#         self.classInfo = dict(zip(self.classNames, self.classIDs))
        
        
        ## Dictionary Parameter
        self.Npatches = 250000;             # Number of patches
        self.Ncentroids = 200;              # Number of centroids
        self.rfSize = 6;                    # Receptor Field Size (i.e. Patch Size)
        self.kmeansIterations = 100         # Iterations for kmeans centroid computation
        self.whitening = True               # Whether to use whitening
        self.normContrast = True            # Whether to normalize patches for contrast
        self.minibatch = True               # Use minibatch to train SVM 
        self.MIN_PATCH_VAR = float(38)/255  # Minimum Patch Variance for accepting as potential centroid (empirically set to about 25% quartile of var)
        self.MAX_TRY = 30                   # Maximum number of try to find a qualify patch from an image
        self.kmeansIterations = 50      
        
        ##### Training #####
        if self.isTrain:
            ## New Model Name
            self.modelName = 'nClass_%d_' % len(self.classNames)
            ## Corpus Path
            self.trainCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat0124_sub"
#             self.trainCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat0124"
            ## Model Saving Path
            self.modelSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model/Classifier"
#             self.modelSavingPath = "/home/ec2-user/VizioMetrics/Model/Classifier"
            ## New Model Path
            self.modelPath = Common.getModelPath(self.modelSavingPath, self.modelName)
        
        ##### Testing ######
        if self.isTest:
            ## Model ID
            self.modelName = 'nClass_5_2015-03-09_CPR'
            ## Model Saving Path
            self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model/Classifier'
#             self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model/Classifier'
            ## Corpus Path
            self.testCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat014_test"
#             self.testCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat014_test"
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
            self.modelName = 'nClass_5_2015-03-09_CPR'
            ## Model Saving Path
            self.modelSavingPath = '/Users/sephon/Desktop/Research/VizioMetrics/Model/Classifier'
#             self.modelSavingPath = '/home/ec2-user/VizioMetrics/Model/Classifier'
            ## Corpus Path
            self.classifyCorpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Classifier/testCorpus'
#             self.classifyCorpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/test_corpus/large_corpus'
#             self.classifyCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat014_test"
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
        
    