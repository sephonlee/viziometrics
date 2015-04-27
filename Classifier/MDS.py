# Author: Nelle Varoquaux <nelle.varoquaux@gmail.com>
# Licence: BSD

print(__doc__)
import numpy as np

from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection

from sklearn import manifold
from sklearn.metrics import euclidean_distances
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import *

# Main program to train model
from Options import *
from Models import *
from DataManager import *
from Dictionary import *


def showMDSAnalysis(X, q_class, n_components):
    similarities = euclidean_distances(X)
    print 'similarities...'
#     similarities = 1 - chi2_kernel(X, gamma=.5)
    print 'mds...'
    mds = manifold.MDS(n_components=2, max_iter=3000, eps=1e-9,
                   dissimilarity="precomputed", n_jobs=1)
    pos = mds.fit(similarities).embedding_
     
#     print 'nmds...'
#     nmds = manifold.MDS(n_components=2, metric=False, max_iter=3000, eps=1e-12,
#                         dissimilarity="precomputed", n_jobs=1,
#                         n_init=1)
#     npos = nmds.fit_transform(similarities, init=pos)
    
    clf = PCA(n_components = 2)
    
    X = clf.fit_transform(X)

    pos = clf.fit_transform(pos)
 
#     npos = clf.fit_transform(npos)
    
    
    fig = plt.figure(1)
    ax = plt.axes([0., 0., 1., 1.])
    
    color = 'wmgbr'
    mark = 'ox+Ds'
    c = 0
    start = 0
    for n in q_class:
        end = start + n
        print start, end
#         plt.scatter(X[i:i+50, 0], X[i:i+50, 1], c=color[c], marker = mark[c])
        plt.scatter(pos[start:end, 0], pos[start:end, 1], s=20, c=color[c], marker = mark[c])
#         plt.scatter(npos[i:i+50, 0], npos[i:i+50, 1], s=20, c=color[c], marker = mark[c])
#         plt.legend(('True position', 'MDS', 'NMDS'), loc='best')
        c += 1
        start = end
      
    plt.legend(('equation', 'photo','scheme', 'table', 'visualization'), loc='best')
     
    similarities = similarities.max() / similarities * 100
    similarities[np.isinf(similarities)] = 0
      
    plt.show()
    
    return pos
    
if __name__ == '__main__':
    
#     # Train Model
    Opt_train = Option(isTrain = True)
    Opt_train.saveSetting()
    ImageLoader_train = ImageLoader(Opt_train)
        
    allImData, allLabels, allCatNames, newClassNames = ImageLoader_train.loadTrainDataFromLocalClassDir(Opt_train.trainCorpusPath)      
#     Opt_train.updateClassNames(newClassNames)
#        
#     Dictionary_train = DictionaryExtractor(Opt_train)
#     dicPath = Dictionary_train.getLocalDictionaryPath(allImData, allLabels)
#                  
#     FD_train= FeatureDescriptor(dicPath)
#     X = FD_train.extractFeatures(allImData, 1)
#     print X.shape
#     y = allCatNames
     
#     np.save('/Users/sephon/Desktop/Research/VizioMetrics/MDS/feature_0124_sub.npy', X)
#     np.save('/Users/sephon/Desktop/Research/VizioMetrics/MDS/name_0124_sub.npy', y)
#     X = np.load('/Users/sephon/Desktop/Research/VizioMetrics/MDS/feature_0124_sub.npy')
#     y = np.load('/Users/sephon/Desktop/Research/VizioMetrics/MDS/name_0124_sub.npy')

    
#     q_class = [300, 300, 300, 300, 300]
#     pos = showMDSAnalysis(X, q_class, 2)
    
#     outPath = '/Users/sephon/Desktop/Research/VizioMetrics/MDS/'
#     Common.saveCSV(outPath, '2Ddata.csv', pos, ('x1', 'x2'), 'wb', True)
#     Common.saveCSV(outPath, 'class.csv', zip(y), ['class'], 'wb', True)
    
    
    
#     SVM_train = SVMClassifier(Opt_train, isTrain = True)
#     SVM_train.trainModel(X, y)
#     print 'Model has been trained'
#     
#     
# 
#     A = [[3,4,1],[2,3,4],[1,6,3],[4,1,2], [9,9,9]]
#     K = chi2_kernel(A, gamma=.5)