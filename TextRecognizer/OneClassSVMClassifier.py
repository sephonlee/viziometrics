import sys
sys.path.append("..")

from Classifier.Models import *

class OneClassSVMClassifier(SVMClassifier):
    
    def trainModel(self, X, y, outModelPath = None):
        
        if outModelPath is None:
            outModelPath = self.Opt.modelPath
            
        print 'Training Model...'
        startTime = time.time()
#         self.classNames = classNames
        # Split into training and test set (e.g., 80/20)
        X_train, X_test, y_train, y_test = cross_validation.train_test_split(X, y, test_size=0.25, random_state=0)
        
        
        clf = svm.OneClassSVM(nu=0.5, kernel="rbf", gamma=0.1)
        clf.fit(X_train)
        y_pred_train = clf.predict(X_train)
        y_pred_test = clf.predict(X_test)
#         y_pred_outliers = clf.predict(X_outliers)
        n_error_train = y_pred_train[y_pred_train == -1].size
        n_error_test = y_pred_test[y_pred_test == -1].size
#         n_error_outliers = y_pred_outliers[y_pred_outliers == 1].size

        print 'n_errpr train', n_error_train, y_train.shape[0]
        print 'n_error test', n_error_test, y_test.shape[0]
        
        # Choose estimator
#         self.estimator = svm.SVC(kernel = 'linear', probability = True)
#         self.estimator = svm.OneClassSVM()
        
        # Choose cross-validation iterator
#         cv = cross_validation.ShuffleSplit(X_train.shape[0], n_iter=10, test_size=0.25, random_state=0)
        
        
        # Tune the hyperparameters
#         gammas = np.logspace(-6, -1, 10)
#         self.classifier = grid_search.GridSearchCV(estimator=self.estimator, cv=cv, param_grid=dict(gamma=gammas))
        
#         tuned_parameters = [{'kernel': ['rbf', 'linear', 'poly'], 'gamma': [1e-3, 1e-4]},]
#         self.classifier = grid_search.GridSearchCV(estimator=self.estimator, cv=cv, param_grid=tuned_parameters)     
        
        
        
#         # Train the optimized model with the split training set
#         self.classifier.fit(X_train, y_train)
#         self.modelOptimized = True
#         
        # Evaluate Cross-validation model by holdout test data
#         self.evaluateCVModel(X_train, y_train, X_test, y_test, path = outModelPath)
#           
#         print 'here'
#         # Train final model with the full training set
#         print 'Train final model with the full training set...'
#         self.classifier.fit(X, y)
#         self.modelTrained = True
#         print 'Tuned Model: Full training data accuracy:', self.classifier.score(X, y)
#         scores = self.getTenFoldValidation(X, y)
#         print "Tuned Model: 10-Fold cross-validation accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2)
#          
#         result = [['Cross-Validation', 'Full Data', 'Accuracy:', scores.mean(), scores.std()*2]]
#          
#         Common.saveCSV(outModelPath, 'model_evaluation', result, mode = 'ab')
#          
#         endTime = time.time()
#         print 'Complete training model in ',  endTime - startTime, 'sec\n'
        
        return self.saveSVMModel(path = outModelPath) 
