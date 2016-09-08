"""Finds and reads text in an image.

Usage:
  main.py [IN] [OUT] [height] [width] [--r] [--ar] [--fs] [--debug]
  main.py (-h | --help)
  main.py --version

Options:
  --debug           Write debug output.
  height            Final image height (Default: 256)
  width             Final image width (Default: 256)
  --r               Resize images (Default: False)
  --ar              Preserve aspect ratio (Default: True)
  --fs               Fix the size when image size < given size (Default: True)
  -h --help         Show this screen.
  --version         Show version.
"""

import logging
import os
import numpy as np
import cv2 as cv
from docopt import docopt
import matplotlib.pyplot as plt


VALID_FORMAT = ["jpeg", "jpg", "png", "bmp"]

TRAIN_SET_PROP = 0.80
VALIDATE_SET_PROP = 0.10
TEST_SET_PROP = 0.10

FIX_ASPECT_RATIO = True
RESIZE_IMAGE = False
FIX_SMALL_SIZE = True

FINAL_IMAGE_HEIGHT = 256
FINAL_IMAGE_WIDTH = 256


def getFileNameAndSuffix(filePath):
    filename = filePath.split('/')[-1]
    suffix = filename.split('.')[-1]
    return filename, suffix


def findBackGroundColor(img, sample_width = 3):
    color = ('b','g','r')
    
    max_count = 0
    max_index = 0
    
    for i,col in enumerate(color):
        histr = cv.calcHist([img],[i],None,[256],[0,256])
        plt.plot(histr,color = col)
        plt.xlim([0,256])
        
        if np.max(histr) > max_count:
            max_count = np.max(histr)
            max_index = np.argmax(histr)
        
        print np.argmax(histr), np.max(histr)
        
    print "max_index", max_index
    plt.show()
    
    
def resize(img, target_height, target_width):
    
#     findBackGroundColor(img, sample_width = 3)
    
    if np.max(img) > 1:
        white_pixel_value = 255;
    else:
        white_pixel_value = 1
    
    imDim = img.shape
    if len(imDim) == 3:
        newImg = np.zeros([FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH, 3], dtype = img.dtype) + white_pixel_value
    else:
        newImg = np.zeros([FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH], dtype = img.dtype) + white_pixel_value
            
    origin_height = imDim[0]
    origin_width = imDim[1]
    
    aspect_ratio = float(origin_height) / origin_width;

    
    if FIX_SMALL_SIZE and origin_height < target_height and origin_width < target_width:
          
        offset_width = (target_width - origin_width) / 2 
        offset_height = (target_height - origin_height) / 2   
        newImg[offset_height : (offset_height + origin_height), offset_width : (offset_width + origin_width), :] = img
      
    else:
        if aspect_ratio > 1: # tall
            tmp_width = int(target_height / aspect_ratio)
            offset_width = (target_width - tmp_width) / 2
              
            tmp_img = cv.resize(img, (tmp_width, target_height))
            if len(imDim) == 3:
                newImg[:, offset_width : (offset_width + tmp_width), :] = tmp_img
            else:
                newImg[:, offset_width : (offset_width + tmp_width)] = tmp_img
              
        else: # wide
            tmp_height = int(target_width * aspect_ratio)            
            offset_height = (target_height - tmp_height) / 2
              
            tmp_img = cv.resize(img, (target_width, tmp_height))
            
            if len(imDim) == 3:
                newImg[offset_height : (offset_height + tmp_height), :, :] = tmp_img
            else:
                newImg[offset_height : (offset_height + tmp_height), :] = tmp_img
                
#     plt.imshow(newImg, interpolation = 'bicubic')
#     plt.show()    
#     
    return newImg
    
def convert_image():
    
    if not os.path.isdir(OUTPUT_PATH):
        os.mkdir(OUTPUT_PATH)
    
    cat_dict = {}
    cat_id = 1
    filename = []
    dest_name = []
    y = []
    try:
        for dirPath, dirNames, fileNames in os.walk(IN_DIR_PATH):   
            for f in fileNames:
                fname, suffix = getFileNameAndSuffix(f)
                if suffix in VALID_FORMAT:
                    cat = dirPath.split('/')[-1]
                    if not cat_dict.has_key(cat):
                        cat_dict[cat] = cat_id
                        cat_id += 1
    
                    filename.append(os.path.join(dirPath,f))
                    dest_name.append(os.path.join(OUTPUT_PATH,cat,f))
                    y.append(cat_dict[cat])
    except:
        print "The given path is not correct"
    
    for key in cat_dict:
        if not os.path.isdir(os.path.join(OUTPUT_PATH,key)):
            os.mkdir(os.path.join(OUTPUT_PATH,key))
    
    num_file = len(filename)
    for i in range(0, num_file):
        
        print "(%d/%d) Converting %s..." %(i+1, num_file, filename[i])
        img = cv.imread(filename[i])
        newImg = resize(img, FINAL_IMAGE_HEIGHT, FINAL_IMAGE_WIDTH)
        cv.imwrite(dest_name[i], newImg)
        
        print "Finished Converting"


if __name__ == '__main__':
    arguments = docopt(__doc__, version='ImageSizeConverter 0.1')
    
#     IN_DIR_PATH = os.getcwd()
    IN_DIR_PATH = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool/"
    OUTPUT_PATH = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/DeepLearning/20160511_all_labelled_images_0214_randomsub_onlinetool_resized/"
#     OUTPUT_PATH = os.getcwd()
    
    if arguments['--debug']:
        logging.basicConfig(level=logging.DEBUG)
        DEBUG = True
     
    if arguments['--r']:
        RESIZE_IMAGE = arguments['--r']
           
    if arguments['--ar']:
        FIX_ASPECT_RATIO = arguments['--ar']
        
    if arguments['--fs']:
        FIX_SMALL_SIZE = arguments['--fs']
        
    if arguments['IN']:
        IN_DIR_PATH = arguments['IN']
        
    if arguments['OUT']:
        OUTPUT_PATH = arguments['OUT']
        
    if arguments['height']:
        FINAL_IMAGE_HEIGHT = arguments['height']
        
    if arguments['width']:
        FINAL_IMAGE_WIDTH = arguments['width']
      
    print "Directory:"
    print "IN_DIR_PATH:", IN_DIR_PATH      
    print "OUTPUT_PATH:", OUTPUT_PATH 
      
    print ""
    print "Setting:"
    print "RESIZE_IMAGE:", RESIZE_IMAGE
    print "FIX_ASPECT_RATIO:", FIX_ASPECT_RATIO      
    print "FIX_SMALL_SIZE:", FIX_SMALL_SIZE      
    print "FINAL_IMAGE_HEIGHT:", FINAL_IMAGE_HEIGHT      
    print "FINAL_IMAGE_HEIGHT:", FINAL_IMAGE_HEIGHT      
        
    convert_image()
#     predict_text(arguments['TEXT_MASK'], arguments['IMAGE'],
#                  int(arguments['--thresh']))
