# Author: Jake Vanderplas -- <vanderplas@astro.washington.edu>

from time import time
from DataFileTool.DataFileTool import *
import matplotlib.pyplot as plt
import cv2 as cv
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.ticker import NullFormatter
from mpl_toolkits.mplot3d import Axes3D
from sklearn.neighbors import NearestNeighbors
from sklearn import manifold, datasets


from Classifier import *


def mds_transform(X, n_components, max_iter=100000, n_init=1):
    
    mds = manifold.MDS(n_components, max_iter=max_iter, n_init=n_init)
    return  mds.fit_transform(X)    

def isomap_transform(X, n_components, n_neighbors):
    isomap = manifold.Isomap(n_neighbors, n_components)
    return isomap.fit_transform(X)

def get_nearest_neighbor(k, X, data):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X) 
    return neigh.kneighbors(data, return_distance=False)
    
def get_nearest_neighbor_graph(k, X):
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X) 
    A = neigh.kneighbors_graph(X)
    distances, indices = neigh.kneighbors(X)
    return distances, indices, A

def get_nearest_neighbor_(k, X):
    neigh = NearestNeighbors(n_neighbors=k, algorithm='ball_tree')
    neigh.fit(X)
    distances, indices = neigh.kneighbors(X)


def scatter_plot(Y, label, colors, label_dic):
    Y_group = [0]
    start_label = label[0]
    
    num = Y.shape[0]
      
    # Find boundary
    for i in range(0, num):
        if label[i] != start_label:
            Y_group.append(i)
            start_label = label[i]
    Y_group.append(num)
     
    
    fig = plt.figure()
    ax = fig.add_subplot(1,1,1)#, projection='3d')
      
    for i in range(0, len(Y_group)-1):
        start = Y_group[i]
        end = Y_group[i+1]
          
#         print "i: %d, %s, start:%d, end:%d" %(i, Classifier.label_dic[i], start, end)
          
        data = Y[start:end, :]
        color = colors[i]
        plt.scatter(data[:, 0], data[:, 1], c=color, cmap=plt.cm.Spectral, label = label_dic[i])
          
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    plt.legend(loc=3)
    plt.axis('tight')
    plt.show()

def export_data(data, label, label_dic, csvSavingPath, csvFilename):
    # Save csv
    header = ['x1', 'x2', 'file_path', 'class']
    DataFileTool.saveCSV(csvSavingPath, csvFilename, header = header, mode = 'wb', consoleOut = False)
    
    for i in range(0, data.shape[0]):
        outcsv = open(os.path.join(csvSavingPath, csvFilename + '.csv'), 'ab')
        writer = csv.writer(outcsv, dialect = 'excel')
        writer.writerow([data[i, 0], data[i, 1], file_list[i],  label_dic[int(label[i])]])
        outcsv.flush()
        outcsv.close()
        
    print "Data saved in %s" %(os.path.join(csvSavingPath, csvFilename + '.csv'))
        
# if __name__ == '__main__':
    
Axes3D

caffe_model_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/models/snapshot_iter_39024.caffemodel'
deploy_file_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/models/deploy.prototxt' # Network definition file
mean_file_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/mean.binaryproto' # Mean image file
syntax_file_path ='/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/data/category.txt'

image_folder_path = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/test"
    

lmdb_test = "/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/viziometrics_test_lmdb/"

Classifier = CNNClassifier(caffe_model_path, deploy_file_path, mean_file_path, syntax_file_path)
lmdb_cursor, datum = CNNClassifier.loadLmdb(lmdb_test)


file_list = []
X = np.zeros([1878,4096])
label = np.zeros([1878])
print X.shape

count = 0
for key, value in lmdb_cursor:
    datum.ParseFromString(value)
    data = caffe.io.datum_to_array(datum)
    img = CNNClassifier.swapOpenCVImg(data)

    X[count, :] = Classifier.getFeature(img, "fc7")
    label[count] = datum.label
    file_list.append(os.path.join("imgs/test", CNNClassifier.getFileNameFromLmdbKey(key)))
    
    count += 1
    
#     if count > 100:
#         break;




# print X    
# print X.shape 
# print label
# 
n_components = 2
# 
colors =np.array( ["salmon",  "yellow", "darkslateblue",  "red", "blue", "blueviolet", "black", "lightgrey"])
# 
# order = np.argsort(label)
# 
# X = X[order]
# label = label[order]
# file_list = np.asarray(file_list)[order]
# 
# Y = mds_transform(X, n_components)
# scatter_plot(Y, label, colors, Classifier.getLabelDic())
# 
# 
# # Save csv
# csvSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/data/"
# csvFilename = "test_data_mds"
# export_data(Y, label, Classifier.getLabelDic(), csvSavingPath, csvFilename)
# 
 
n_neighbors = 10
Y = isomap_transform(X, n_components, n_neighbors)
# scatter_plot(Y, label, colors, Classifier.getLabelDic())

distances, indices, A = get_nearest_neighbor_graph(7, X)
# print neighbors_matrix
# print neighbors_matrix.toarray()

for i in range(0, X.shape[0]):
    print 
    print "Image: ", file_list[i]
    filename = os.path.join(image_folder_path, file_list[i].split("/")[-1])
    plt.imshow(cv.imread(filename), interpolation = 'bicubic')
    plt.show()
        
    print indices[i, :].tolist()
    for n in indices[i, :].tolist():
        print "Neighbor: ", file_list[n]
    
        filename = os.path.join(image_folder_path, file_list[n].split("/")[-1])
        plt.imshow(cv.imread(filename), interpolation = 'bicubic')
        plt.show()

#     
#     row = np.squeeze(np.array(neighbors_matrix[i, :]))
#     print neighbors_matrix[i, :]
#     print row.tolist()
#     neighbors = np.where( row == 1.0 )
#     print "neighbors:", neighbors
#     for j in range(0, neighbors.shape[0]):
#         print neighbors[j],  file_list[j]
    
    
# 
# 
# csvSavingPath = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/data/"
# csvFilename = "test_data_isomap"
# export_data(Y, label, Classifier.getLabelDic(), csvSavingPath, csvFilename)





# # Next line to silence pyflakes. This import is needed.
# Axes3D
# 
# n_points = 1000
# X, color = datasets.samples_generator.make_s_curve(n_points, random_state=0)
# n_neighbors = 10
# n_components = 2
# 
# fig = plt.figure(figsize=(15, 8))
# plt.suptitle("Manifold Learning with %i points, %i neighbors"
#              % (1000, n_neighbors), fontsize=14)
# 
# try:
#     # compatibility matplotlib < 1.0
#     ax = fig.add_subplot(251, projection='3d')
#     ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=color, cmap=plt.cm.Spectral)
#     ax.view_init(4, -72)
# except:
#     ax = fig.add_subplot(251, projection='3d')
#     plt.scatter(X[:, 0], X[:, 2], c=color, cmap=plt.cm.Spectral)
# 
# methods = ['standard', 'ltsa', 'hessian', 'modified']
# labels = ['LLE', 'LTSA', 'Hessian LLE', 'Modified LLE']
# 
# for i, method in enumerate(methods):
#     t0 = time()
#     Y = manifold.LocallyLinearEmbedding(n_neighbors, n_components,
#                                         eigen_solver='auto',
#                                         method=method).fit_transform(X)
#     t1 = time()
#     print("%s: %.2g sec" % (methods[i], t1 - t0))
# 
#     ax = fig.add_subplot(252 + i)
#     plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
#     plt.title("%s (%.2g sec)" % (labels[i], t1 - t0))
#     ax.xaxis.set_major_formatter(NullFormatter())
#     ax.yaxis.set_major_formatter(NullFormatter())
#     plt.axis('tight')
# 
# t0 = time()
# Y = manifold.Isomap(n_neighbors, n_components).fit_transform(X)
# t1 = time()
# print("Isomap: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(257)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("Isomap (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# 
# 
# t0 = time()
# mds = manifold.MDS(n_components, max_iter=100, n_init=1)
# Y = mds.fit_transform(X)
# t1 = time()
# print("MDS: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(258)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("MDS (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# 
# 
# t0 = time()
# se = manifold.SpectralEmbedding(n_components=n_components,
#                                 n_neighbors=n_neighbors)
# Y = se.fit_transform(X)
# t1 = time()
# print("SpectralEmbedding: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(259)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("SpectralEmbedding (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# 
# t0 = time()
# tsne = manifold.TSNE(n_components=n_components, init='pca', random_state=0)
# Y = tsne.fit_transform(X)
# t1 = time()
# print("t-SNE: %.2g sec" % (t1 - t0))
# ax = fig.add_subplot(2, 5, 10)
# plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
# plt.title("t-SNE (%.2g sec)" % (t1 - t0))
# ax.xaxis.set_major_formatter(NullFormatter())
# ax.yaxis.set_major_formatter(NullFormatter())
# plt.axis('tight')
# 
# plt.show()