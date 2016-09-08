#!/usr/bin/env python
# Copyright (c) 2015, NVIDIA CORPORATION.  All rights reserved.

"""
Classify an image using individual model files
Use this script as an example to build your own tool
"""

import argparse
import os
import time
import cv2
import PIL.Image
import numpy as np
os.environ['GLOG_minloglevel'] = '2' # Suppress most caffe output

from matplotlib import pyplot as plt
from google.protobuf import text_format
from caffe.proto import caffe_pb2

import lmdb
import caffe




def forward_pass(images, net, transformer, batch_size=1):
    """
    Returns scores for each image as an np.ndarray (nImages x nClasses)
    Arguments:
    images -- a list of np.ndarrays
    net -- a caffe.Net
    transformer -- a caffe.io.Transformer
    Keyword arguments:
    batch_size -- how many images can be processed at once
        (a high value may result in out-of-memory errors)
    """
    caffe_images = []
    for image in images:
        if image.ndim == 2:
            caffe_images.append(image[:,:,np.newaxis])
        else:
            caffe_images.append(image)

    caffe_images = np.array(caffe_images)

    dims = transformer.inputs['data'][1:]

    scores = None
    for chunk in [caffe_images[x:x+batch_size] for x in xrange(0, len(caffe_images), batch_size)]:
        new_shape = (len(chunk),) + tuple(dims)
        if net.blobs['data'].data.shape != new_shape:
            net.blobs['data'].reshape(*new_shape)
        for index, image in enumerate(chunk):
            image_data = transformer.preprocess('data', image)
            net.blobs['data'].data[index] = image_data
        output = net.forward()[net.outputs[-1]]
        if scores is None:
            scores = output
        else:
            scores = np.vstack((scores, output))
        print 'Processed %s/%s images ...' % (len(scores), len(caffe_images))

    return scores






class CNNClassifier():
    
    net = None              #CNN Net
    transformer = None      #CNN transformer
    label_dic = None        #Label Dictionary, key = label_id, value = label_name
    
    def __init__(self, caffe_model_path, deploy_file_path, mean_file_path, syntax_file_path= None):
        
        self.getNet(caffe_model_path, deploy_file_path, use_gpu=False)
        
        self.getTransformer(mean_file_path)
        self.syntax_file_path = syntax_file_path
        
        if syntax_file_path:
            self.loadLabelName(syntax_file_path)
        
    
    def getNet(self, caffe_model_path, deploy_file_path, use_gpu=False):
        """
        Returns an instance of caffe.Net
        Arguments:
        caffemodel -- path to a .caffemodel file
        deploy_file -- path to a .prototxt file
        Keyword arguments:
        use_gpu -- if True, use the GPU for inference
        """
        if use_gpu:
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()
    
        # load a new model
        self.net = caffe.Net(deploy_file_path, caffe_model_path, caffe.TEST)

    def getTransformer(self, mean_file_path):
            # set mean pixel
        self.transformer = caffe.io.Transformer({'data': self.net.blobs['data'].data.shape})
        
        with open(mean_file_path) as infile:
            blob = caffe_pb2.BlobProto()
            blob.MergeFromString(infile.read())
            if blob.HasField('shape'):
                blob_dims = blob.shape
                assert len(blob_dims) == 4, 'Shape should have 4 dimensions - shape is "%s"' % blob.shape
            elif blob.HasField('num') and blob.HasField('channels') and \
                    blob.HasField('height') and blob.HasField('width'): 
                blob_dims = (blob.num, blob.channels, blob.height, blob.width)
            else:
                raise ValueError('blob does not provide shape or 4d dimensions')
            pixel = np.reshape(blob.data, blob_dims[1:]).mean(1).mean(1)
            self.transformer.set_mean('data', pixel)
        
        self.transformer.set_transpose('data', (2,0,1))
        self.transformer.set_raw_scale('data', 255.0)
        self.net.blobs['data'].reshape(1,3,227,227)
        

    def loadLabelName(self, syntax_file_path):
        labels = np.loadtxt(syntax_file_path, str, delimiter='\t')
        self.label_dic = {}
        for label in labels:
            l = label.split(" ")
            self.label_dic[int(l[0])] = l[1]    
        return self.label_dic

    def getLabelDic(self):
        if self.label_dic is None:
            print "label dictionary has not been filled"
        return self.label_dic
    
    def getFeature(self, img, layername):
        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
        self.net.forward()
        return self.net.blobs[layername].data.copy();
    
    def predict(self, img):

        self.net.blobs['data'].data[...] = self.transformer.preprocess('data', img)
        output = self.net.forward()        
        label_id = output['prob'].argmax()
        
        if self.label_dic:
            return label_id, self.label_dic[label_id]
        else:
            return label_id, label_id
    
    @staticmethod
    def swapOpenCVImg(img):
        img = np.transpose(img, (1, 2, 0))
        img = img[:,:,(2,1,0)]
        return img / 255.
    
    @staticmethod
    def cvImRead(filepath):
        return cv2.imread(filepath)
    
    @staticmethod     
    def loadLmdb(lmdbpath):
         
        lmdb_env = lmdb.open(lmdbpath)
        lmdb_txn = lmdb_env.begin()
        lmdb_cursor = lmdb_txn.cursor()
        datum = caffe_pb2.Datum()
         
        return lmdb_cursor, datum
    
    @staticmethod
    def getFileNameFromLmdbKey(key):
        filename = key.split("_")
        filename.pop(0)
        return "_".join(filename)
    
    def evaluateModel(self, lmdbpath, show_iteration = False):
        
        lmdb_cursor, datum = CNNClassifier.loadLmdb(lmdbpath)
        
        confusion_matrix = np.zeros((len(self.label_dic), len(self.label_dic)))
 
        for key, value in lmdb_cursor:
            datum.ParseFromString(value)
            data = caffe.io.datum_to_array(datum)
            img = CNNClassifier.swapOpenCVImg(data)
    
            label_id, label_name = Classifier.predict(img)            
            true_label = datum.label
            true_label_name = self.label_dic[true_label]
            
            confusion_matrix[true_label][label_id] += 1
            
            if show_iteration:
                print "predict label: %s(%d), true label: %s(%d)" %(label_name, label_id, true_label_name, true_label)       

        overall_true_count = 0
        overall_num_image = np.sum(confusion_matrix)
        
        print ""
        for key in self.label_dic:
            true_count = confusion_matrix[key][key]
            overall_true_count += true_count
            all_count = np.sum(confusion_matrix[key, :])
            print "%s recall: %f(%d/%d)" %(self.label_dic[key], float(true_count)/all_count, true_count, all_count)
        
        print ""
        for key in self.label_dic:
            true_count = confusion_matrix[key][key]
            all_count = np.sum(confusion_matrix[:, key])
            print "%s precision: %f(%d/%d)" %(self.label_dic[key], float(true_count)/all_count, true_count, all_count)
        
        print ""
        print "Accuracy: %f(%d/%d)" %(float(overall_true_count)/overall_num_image, overall_true_count, overall_num_image)
        
        return None
    

if __name__ == '__main__':
    script_start_time = time.time()

    caffe_model_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/models/snapshot_iter_39024.caffemodel'
    deploy_file_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/models/deploy.prototxt' # Network definition file
    mean_file_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/mean.binaryproto' # Mean image file
    syntax_file_path ='/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/data/category.txt'
 
    lmdb_test = "/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project_8cat/viziometrics_test_lmdb/"
    
    Classifier = CNNClassifier(caffe_model_path, deploy_file_path, mean_file_path, syntax_file_path)
    Classifier.evaluateModel(lmdb_test, show_iteration = True)

    
#     # and the image you would like to classify.
# MODEL_FILE = '../models/bvlc_reference_caffenet/deploy.prototxt'
# PRETRAINED = '../models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
# IMAGE_FILE = 'images/cat.jpg'

#     caffe.set_mode_cpu()
#     net = caffe.Classifier(caffe_model_path, caffe_model_path,
#                        mean=mean_file_path,
#                        channel_swap=(2,1,0),
#                        raw_scale=255,
#                        image_dims=(256, 256))

#     classify(caffe_model_path, deploy_file_path, image_file_path, mean_file = mean_file_path)

#     parser = argparse.ArgumentParser(description='Classification example - DIGITS')
# 
#     ### Positional arguments
# 
#     parser.add_argument('caffemodel',   help='Path to a .caffemodel')
#     parser.add_argument('deploy_file',  help='Path to the deploy file')
#     parser.add_argument('image',        help='Path to an image')
# 
#     ### Optional arguments
# 
#     parser.add_argument('-m', '--mean',
#             help='Path to a mean file (*.npy)')
#     parser.add_argument('-l', '--labels',
#             help='Path to a labels file')
#     parser.add_argument('--nogpu',
#             action='store_true',
#             help="Don't use the GPU")
# 
#     args = vars(parser.parse_args())
# 
#     image_files = [args['image']]
# 
#     classify(args['caffemodel'], args['deploy_file'], image_files,
#             args['mean'], args['labels'], not args['nogpu'])
# 
#     print 'Script took %s seconds.' % (time.time() - script_start_time,)
    
    
    
    



# # Run the script with anaconda-python
# # $ /home/<path to anaconda directory>/anaconda/bin/python LmdbClassification.py
# import sys
# import numpy as np
# import lmdb
# import caffe
# from collections import defaultdict
# caffe.set_mode_cpu()
# 
# # Modify the paths given below
# deploy_prototxt_file_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project/models/deploy.prototxt' # Network definition file
# caffe_model_file_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project/models/snapshot_iter_4123.caffemodel' # Trained Caffe model file
# test_lmdb_path = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project/viziometrics_test_lmdb/' # Test LMDB database path
# mean_file_binaryproto = '/Users/sephon/Desktop/Research/VizioMetrics/Deep_learning/project/mean.binaryproto' # Mean image file
# 
# # Extract mean from the mean image file
# mean_blobproto_new = caffe.proto.caffe_pb2.BlobProto()
# f = open(mean_file_binaryproto, 'rb')
# mean_blobproto_new.ParseFromString(f.read())
# mean_image = caffe.io.blobproto_to_array(mean_blobproto_new)
# f.close()
# 
# # CNN reconstruction and loading the trained weights
# net = caffe.Net(deploy_prototxt_file_path, caffe_model_file_path, caffe.TEST)
# 
# count = 0
# correct = 0
# matrix = defaultdict(int) # (real,pred) -> int
# labels_set = set()
#  
# lmdb_env = lmdb.open(test_lmdb_path)
# lmdb_txn = lmdb_env.begin()
# lmdb_cursor = lmdb_txn.cursor()
#  
# for key, value in lmdb_cursor:
#     datum = caffe.proto.caffe_pb2.Datum()
#     datum.ParseFromString(value)
#     label = int(datum.label)
#     image = caffe.io.datum_to_array(datum)
#     image = image.astype(np.uint8)
#     
#     print np.asarray([image]).shape
#     print mean_image.shape
#     
#     out = net.forward_all(data=np.asarray([image]) - mean_image)
#     plabel = int(out['prob'][0].argmax(axis=0))
#     count += 1
#     iscorrect = label == plabel
#     correct += (1 if iscorrect else 0)
#     matrix[(label, plabel)] += 1
#     labels_set.update([label, plabel])
#  
#     if not iscorrect:
#         print("\rError: key = %s, expected %i but predicted %i" % (key, label, plabel))
#         sys.stdout.write("\rAccuracy: %.1f%%" % (100.*correct/count))
#         sys.stdout.flush()
#  
# print("\n" + str(correct) + " out of " + str(count) + " were classified correctly")
# print ""
# print "Confusion matrix:"
# print "(r , p) | count"
# for l in labels_set:
#     for pl in labels_set:
#         print "(%i , %i) | %i" % (l, pl, matrix[(l,pl)])