from Options import *
from Classification import *
from DataManager import *
from Dictionary import *

if __name__ == '__main__':
    
    ## Test Training
    Opt_train = Opt(isTrain = True)
    Opt_train.saveSetting()
    ImageLoader_train = ImageLoader(Opt_train)
    allImData, allLabels, allCatNames = ImageLoader_train.loadTrainDataFromLocalClassDir(Opt_train.trainCorpusPath)
     
    Dictionary_train = DictionaryExtractor(Opt_train)
    dicPath = Dictionary_train.getLocalDictionaryPath(allImData, allLabels)
     
    FD_train= FeatureDescriptor(dicPath)
    X = FD_train.extractFeatures(allImData, 1)
    y = allCatNames
           
    SVM_train = SVMClassifier(Opt_train, isTrain = True)
    SVM_train.trainModel(X, y)
      
    modelPath = SVM_train.saveSVMModel()
#     y_pred, y_proba = SVM_train.predict(X)
#     SVM_train.evaluate(y, y_pred, y_proba)
    
   
     
    ## Test Testing
    Opt_test = Opt(isTest = True)
    ImageLoader_test = ImageLoader(Opt_test)
    allImData, allLabels, allCatNames = ImageLoader_test.loadTestDataFromLocalClassDir()
    
#     print allLabels
    
    FD_test = FeatureDescriptor(Opt_test.dicPath)
    X = FD_test.extractFeatures(allImData, 1)
    y = allCatNames

    SVM_test = SVMClassifier(Opt_test, isTrain = False)
    y_pred, y_proba = SVM_test.predict(X)
    SVM_test.evaluate(y, y_pred, y_proba)


#     ## Test Classifying
#     Opt = Opt(isClasify = True)
#     corpusPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/testCorpus"
#     VCLF = VizClassifier(Opt, clf = 'SVM')
#     VCLF.classifyAllImages(corpusPath = corpusPath)