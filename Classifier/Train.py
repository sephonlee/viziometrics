from Options import *
from Models import *
from DataManager import *
from Dictionary import *

if __name__ == '__main__':
    
    ## Train Model
    Opt_train = Option(isTrain = True)
    Opt_train.saveSetting()
    ImageLoader_train = ImageLoader(Opt_train)

    allImData, allLabels, allCatNames, newClassNames = ImageLoader_train.loadTrainDataFromLocalClassDir(Opt_train.trainCorpusPath)      
    Opt_train.updateClassNames(newClassNames)

    Dictionary_train = DictionaryExtractor(Opt_train)
    dicPath = Dictionary_train.getLocalDictionaryPath(allImData, allLabels)
          
    FD_train= FeatureDescriptor(dicPath)
    X = FD_train.extractFeatures(allImData, 1)
    y = allCatNames
                
    SVM_train = SVMClassifier(Opt_train, isTrain = True)
    SVM_train.trainModel(X, y)
    print 'Model has been trained'


