from Options import *
from Models import *
from DataManager import *
from Dictionary import *

if __name__ == '__main__':
    
    # Test Testing
    Opt_test = Option(isTest = True)
    FD = FeatureDescriptor(Opt_test.dicPath)
    SVM_test = SVMClassifier(Opt_test, isTrain = False)
    
    ImageLoader_test = ImageLoader(Opt_test)
    
    
    testFileLists = {
                      "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/data/test.txt" : "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/test",
                      }
    
    categoryFilePath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/data/category.txt"
#     categoryFilePath = "/home/ec2-user/VizioMetrics/Corpus/Classifier/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/data/category.txt"
    
    
    allImData, allLabels, allCatNames, newClassNames = ImageLoader_test.loadTrainDataFromFileList(testFileLists, categoryFilePath=categoryFilePath, mode = "test")
#     allImData, allLabels, allCatNames, classNames = ImageLoader_test.loadTestDataFromLocalClassDir()
    
    
    
    X = FD.extractFeatures(allImData, 1)
    y = allCatNames
#       
#     SVM_test = SVMClassifier(Opt_test, isTrain = False)
    y_pred, y_proba = SVM_test.predict(X)

    # Save confusion matrix
    SVM_test.saveEvaluation(y, y_pred)
    # Save accuracy 
    SVM_test.evaluate(y, y_pred, y_proba, save = True)
