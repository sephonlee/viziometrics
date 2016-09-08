# Main program to train model
import sys
sys.path.append("..")
from Dismantler.Dismantler import *
from Dismantler.TrainCompositeDetector import *

from Options import *
from Models import *
from DataManager import *
from Dictionary import *

Opt_Dmtler = Option_Dismantler(isTrain = False)
Dmtler = Dismantler(Opt_Dmtler)
Opt_CID = Option_CompositeDetector(isTrain = True)
CID = CompositeImageDetector(Opt_CID)
    
def getFeatureByFireLaneMapFromFileList(fileList, showStatus = True):
        
    data = np.zeros([len(fileList), 2 + CID.division[1] * CID.division[0]])
    for (i,filename) in enumerate(fileList):
        
        img = cv.imread(filename)
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        first_vertical, fire_lane_map_vertical, count_standalone_first_vertical = Dmtler.split(img, 0)
        first_horizontal, fire_lane_map_horizontal, count_standalone_first_horizontal = Dmtler.split(img, 1)
        map = fire_lane_map_vertical + fire_lane_map_horizontal
#         Dmtler.showImg(fire_lane_map_vertical)
#         Dmtler.showImg(fire_lane_map_horizontal)
#         Dmtler.showImg(map)
        
        data[i, :] = CID.getFeatureByFireLaneMap(map)
        
        if showStatus:
            print '%d / %d images has been collected' %(i, len(fileList))
        
    print '%d images has been collected.' %len(fileList)
    return data

def normalize(XC):
    FSMean = np.mean(XC, axis = 0)
    FSSd = np.sqrt(np.var(XC, axis = 0) + 0.01)
    XCs = np.divide(XC - FSMean, FSSd)
    return XCs

if __name__ == '__main__':
    
    ## Train Model
    Opt_train = Option(isTrain = True)
    Opt_train.saveSetting()
    ImageLoader_train = ImageLoader(Opt_train)
    

### Test firelane performance
#     countClass = 0
#     for classname in Opt_train.classNames:
#         classPath = os.path.join(Opt_train.trainCorpusPath, classname)
#         fileList = ImageLoader_train.getFileNamesFromPath(classPath)
#         print 'Collecting %d images from %s...' %(len(fileList), classname)
#         if countClass == 0:
#             allERMFeatures = getFeatureByFireLaneMapFromFileList(fileList, showStatus = False)
#         else:
#             allERMFeatures = np.vstack([allERMFeatures, getFeatureByFireLaneMapFromFileList(fileList, showStatus = False)])
#             
#         countClass += 1
#         
#     print allERMFeatures.shape

    allImData, allLabels, allCatNames, newClassNames = ImageLoader_train.loadTrainDataFromLocalClassDir(Opt_train.trainCorpusPath)      
    Opt_train.updateClassNames(newClassNames)
# 
    Dictionary_train = DictionaryExtractor(Opt_train)
    dicPath = Dictionary_train.getLocalDictionaryPath(allImData, allLabels)
#           
    FD_train= FeatureDescriptor(dicPath)
    X = FD_train.extractFeatures(allImData, 1)
#     X = np.hstack([X, allERMFeatures])  // Test firelane feature
#     X = normalize(X)  // Test firelane feature
#     print X.shape  // Test firelane feature
    y = allCatNames
#                 
    Common.saveArff(Opt_train.modelPath, 'train_data.arff', X, y)
    
    SVM_train = SVMClassifier(Opt_train, isTrain = True)
    SVM_train.trainModel(X, y)
    print 'Model has been trained'


