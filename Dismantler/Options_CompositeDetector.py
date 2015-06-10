# All parameter setting
# All commonly-used functions about files
import cPickle as pickle
import os, errno
import datetime
import csv

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
        
        if self.isTrain or self.isClassify:
            print 'Warning! Must select either training mode or classifying mode'
        
        ## Classifiers
        self.availableClassifiers = ['SVM', 'CNN']
        self.activatedClassifiers = 'SVM'
        self.tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-4], 'C': [1, 10, 100, 1000]},]

        ## S3 Data Read Parameter
#         self.keyPath = '/Users/sephon/Desktop/Research/VizioMetrics/keys.txt'
        self.keyPath = '/home/ec2-user/VizioMetrics/keys.txt'
        self.host = 'escience.washington.edu.viziometrics'

        ## Data Read Parameter
        self.finalDim = [128, 128, 1]    # Final image dimensions
        self.preserveAspectRatio = True
#         self.Ntrain = 1                  #/ Number of training images per category
#         self.Ntest = 1                   #/ Number of test images per category
        self.validImageFormat = ['jpg', 'tif', 'bmp', 'png', 'tiff']    # Valid image formats
        self.validMinKeySize = 100000000
#         self.classNames = ['electrophoresis', 'equation', 'composite', 'photo','scheme', 'table', 'visualization']
#         self.classNames = ['bar', 'boxplot', 'heatmap', 'line', 'pie', 'scatter']
        self.classNames = ['single', 'composite']
        self.classNames = sorted(self.classNames)
        self.classIDs = range(1, len(self.classNames)+1)        # Start from 1
        self.classInfo = dict(zip(self.classNames, self.classIDs))
        
        ## Descriptor Parameter
        self.num_cut = 12
        self.offset_dim = (400, 400)
        
        ## SVM Classifier Parameter
        
        
        ##### Training #####
        if self.isTrain:
            ## New Model Name
            self.modelName = 'nClass_%d_' % len(self.classNames)
            ## Corpus Path
            self.trainCorpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/CNN/CNN_decomposite"
#             self.trainCorpusPath = "/home/ec2-user/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat014"
            ## Model Saving Path
            self.modelSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Model/Dismantler"
#             self.modelSavingPath = "/home/ec2-user/VizioMetrics/Model/Classifier"
            ## New Model Path
            self.modelPath = Common.getModelPath(self.modelSavingPath, self.modelName)
        
        ##### Testing ######
        if self.isTest:
            ## Model ID
            self.modelName = 'nClass_4_2015-02-19_CPR'
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
            self.modelName = 'nClass_4_2015-02-19_CPR'
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
        return Common.makeDir(path, dirName)
    
    
    @staticmethod
    def getFileNameAndSuffix(filePath):
        filename = filePath.split('/')[-1]
        suffix = filename.split('.')[1]
        return filename, suffix
    
        # File format: paper_id_image_id.{png, jpg,....}
    @ staticmethod
    def getIDsFromFilename(filename):
        filename = filename.split('.')
        filename = filename[0].split('_')
        return filename[1], filename[3]
    
    @ staticmethod
    def saveCSV(path, filename, content = None, header = None, mode = 'wb', consoleOut = True):

        if consoleOut:
            print 'Saving image information...'
        filePath = os.path.join(path, filename) + '.csv'
        with open(filePath, mode) as outcsv:
            writer = csv.writer(outcsv, dialect='excel')
            if header is not None:
                writer.writerow(header)
            if content is not None:
                for c in content:
                    writer.writerow(c)
        
        if consoleOut:  
            print filename, 'were saved in', filePath, '\n'
    