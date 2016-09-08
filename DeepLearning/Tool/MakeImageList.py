"""Finds and reads text in an image.

Usage:
  main.py [IN] [OUT] [--debug]
  main.py (-h | --help)
  main.py --version

Options:
  --debug           Write debug output.
  IN                Path to Image Directory
  OUT               Path to File Output
  -h --help         Show this screen.
  --version         Show version.
"""

import logging
import os
import numpy as np
from shutil import copyfile
from docopt import docopt


VALID_FORMAT = ["jpeg", "jpg", "png", "bmp"]

TRAIN_SET_PROP = 0.80
VALIDATE_SET_PROP = 0.10
TEST_SET_PROP = 0.10



def getFileNameAndSuffix(filePath):
    filename = filePath.split('/')[-1]
    suffix = filename.split('.')[-1]
    return filename, suffix

def make_list():

    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH);
    else:
        print"Warning: %s existed" %OUTPUT_PATH
        print
        
    cat_dict = {}
    cat_id = 0
    x = []
    y = []
    path = []
    try:
        for dirPath, dirNames, fileNames in os.walk(IN_DIR_PATH):   
            for f in fileNames:
                fname, suffix = getFileNameAndSuffix(f)
                if suffix in VALID_FORMAT:
                    cat = dirPath.split('/')[-1]
                    if not cat_dict.has_key(cat):
                        cat_dict[cat] = cat_id
                        cat_id += 1
    
                    x.append(f)
                    y.append(cat_dict[cat])
                    path.append(os.path.join(dirPath,f))
    except:
        print "The given path is not correct"
                
    data_size = len(x)
    arr = np.arange(data_size)
    np.random.shuffle(arr)
    x = np.array(x)[arr][::-1]
    y = np.array(y)[arr][::-1]
    path = np.array(path)[arr][::-1]
    
    validate_start = 0
    validate_end = int(data_size * (VALIDATE_SET_PROP))
    
    test_start = validate_end
    test_end = int(data_size * (VALIDATE_SET_PROP + TEST_SET_PROP))
    
    train_start = test_end
    train_end = int(data_size * (VALIDATE_SET_PROP + TEST_SET_PROP + TRAIN_SET_PROP)) 
 
#     x_train = x[train_start:]
#     y_train = y[train_start:]
#     order = np.argsort(y_train)
#     x_train = x_train[order]
#     y_train = y_train[order]

 
    print "Generating txt files..."
    print ""   
     
    data_list_path = os.path.join(OUTPUT_PATH, "data")
    if not os.path.isdir(data_list_path):
        os.mkdir(data_list_path);
    else:
        print"Warning: %s existed" %data_list_path
        print
        
    with open(os.path.join(data_list_path, "validate.txt"), "wb") as text_file: 
        saving_path = os.path.join(OUTPUT_PATH, "validate")
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)  
        for i in range(validate_start, validate_end):
            text_file.write("%s %d \n" %(x[i], y[i]))
            copyfile(path[i], os.path.join(saving_path, x[i]))
        print "validating image list (%d) is saved in %s" %(validate_end-validate_start,os.path.join(OUTPUT_PATH, "validate.txt"))
              
    with open(os.path.join(data_list_path, "test.txt"), "wb") as text_file:   
        saving_path = os.path.join(OUTPUT_PATH, "test")
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path)
        for i in range(test_start, test_end):
            text_file.write("%s %d \n" %(x[i], y[i]))
            copyfile(path[i], os.path.join(saving_path, x[i]))
        print "testing image list (%d) is saved in %s" %(test_end - test_start, os.path.join(OUTPUT_PATH, "test.txt"))
  
    with open(os.path.join(data_list_path, "train.txt"), "wb") as text_file:
        saving_path = os.path.join(OUTPUT_PATH, "train")
        if not os.path.isdir(saving_path):
            os.mkdir(saving_path);
        for i in range(train_start, train_end):
            text_file.write("%s %d \n" %(x[i], y[i]))
            copyfile(path[i], os.path.join(saving_path, x[i]))
        print "training image list (%d) is saved in %s" %(train_end-train_start,os.path.join(OUTPUT_PATH, "train.txt"))
           
    with open(os.path.join(data_list_path, "category.txt"), "wb") as text_file:   
        for key, val in cat_dict.iteritems():
            text_file.write("%d %s \n" %(val, key))
        print "training image list is saved in %s" %os.path.join(OUTPUT_PATH, "category.txt")
     
    print ""        
    print "Done!"
    


if __name__ == '__main__':
    arguments = docopt(__doc__, version='CaffeFileMaker 0.1')
    
#     IN_DIR_PATH = os.getcwd()
    IN_DIR_PATH = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized/"
    OUTPUT_PATH = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized_for_caffe/"
#     OUTPUT_PATH = os.getcwd()
    
    if arguments['--debug']:
        logging.basicConfig(level=logging.DEBUG)
        DEBUG = True
        
    if arguments['IN']:
        IN_DIR_PATH = arguments['IN']
        
    if arguments['OUT']:
        OUTPUT_PATH = arguments['OUT']

    make_list()
#     predict_text(arguments['TEXT_MASK'], arguments['IMAGE'],
#                  int(arguments['--thresh']))
