from Options import *
from Models import *
from DataManager import *
from Dictionary import *

if __name__ == '__main__':
    
    # Test Testing
    Opt_test = Option(isTest = True)
    ImageLoader_test = ImageLoader(Opt_test)
    allImData, allLabels, allCatNames, classNames = ImageLoader_test.loadTestDataFromLocalClassDir()
         
    FD_test = FeatureDescriptor(Opt_test.dicPath)
    X = FD_test.extractFeatures(allImData, 1)
    y = allCatNames
     
    SVM_test = SVMClassifier(Opt_test, isTrain = False)
    y_pred, y_proba = SVM_test.predict(X)
    SVM_test.evaluate(y, y_pred, y_proba)
    SVM_test.saveEvaluation(y, y_pred)
