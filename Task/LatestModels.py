import sys
sys.path.append("..")

from Classifier.Options import *
from Classifier.Dictionary import *
from Classifier.Models import *
# from Classifier.DataManager import *
from Classifier.DBManager import *

from Dismantler.Options_Dismantler import *
from Dismantler.Dismantler import * 





EC2 = False

if EC2:
    # Model Path
    CCLF_PATH = '/home/ec2-user/VizioMetrics/Model/Classifier/nClass_5_2015-03-09_CPR'
    DMTLER_PATH = '/home/ec2-user/VizioMetrics/Model/Dismantler/dismantler_matsplit_matsvm_ceil_latest'
    CPSD_PATH = '/home/ec2-user/VizioMetrics/Model/Dismantler/composite_detector_firelanemap'
    
    # Data Path
    KEYPATH = '/home/ec2-user/VizioMetrics/keys.txt'
    HOST = 'escience.washington.edu.viziometrics'
else:
    # Model Path
    CCLF_PATH = '/Users/sephon/Desktop/Research/VizioMetrics/Model/Classifier/nClass_5_2015-03-09_CPR'
    DMTLER_PATH = '/Users/sephon/Desktop/Research/VizioMetrics/Model/Dismantler/dismantler_matsplit_matsvm_ceil_latest'
    CPSD_PATH = '/Users/sephon/Desktop/Research/VizioMetrics/Model/Dismantler/composite_detector_firelanemap'
    
    # Data Path
    KEYPATH = '/Users/sephon/Desktop/Research/VizioMetrics/keys.txt'
    HOST = 'escience.washington.edu.viziometrics'
    

## Global Object   
OPT_CCLF = Option(isClassify = True)
OPT_DMTLER = Option_Dismantler(isTrain = False)
OPT_CPSD = Option_CompositeDetector(isTrain = False)
                                    
## Boxes
FD = FeatureDescriptor(CCLF_PATH)
CCLF = SVMClassifier(OPT_CCLF, clfPath = CCLF_PATH)
CIL = CloudImageLoader(OPT_CCLF, keyPath = KEYPATH, host = HOST)
DMTLER = Dismantler(OPT_DMTLER, auxClfPath = DMTLER_PATH)
CPSD = CompositeImageDetector(OPT_CPSD, modelPath = CPSD_PATH)
print CPSD.Classifier.classifier.best_estimator_