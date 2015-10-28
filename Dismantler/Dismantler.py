import sys
sys.path.append("..")
from Classifier.Models import SVMClassifier
from Classifier.DataManager import *

from ete2 import Tree
from ete2 import TreeNode
from matplotlib import pyplot as plt
from sets import Set
from random import randint

# import matplotlib.cm as cm
# import random
import math
import time
import cv2 as cv
import numpy as np

from Options_Dismantler import *



class CompositeImageDetector():
    Opt = None
    Classifier = None
    finalDim = None
    num_cut = None
    offset_dim = None
    
    def __init__(self, Opt, modelPath = None):
        self.Opt = Opt
        if modelPath is None:
            self.modelPath = Opt.modelPath 
        else:
            self.modelPath = modelPath
            
        self.num_cut = Opt.num_cut
        self.offset_dim = Opt.offset_dim
        self.thresholds = Opt.thresholds
        self.division = Opt.division
        
        self.loadSVMClassifier()
        self.loadDescriptorParams()
            
    def loadSVMClassifier(self):
        try:
            self.Classifier = SubImageClassifier(self.Opt, isTrain= False, clfPath = self.modelPath)
            print 'SubImage Classifier ready. \n'
        except:
            print 'Require valid sub-image classifier. \n'  
    
    def loadDescriptorParams(self):
        print os.path.join(self.modelPath, 'descriptorParam.npz')
        try:
            path = os.path.join(self.modelPath, 'descriptorParam.npz')
            dictionary = np.load(path)
            print 'Descriptor Parameters loaded from', path
            self.num_cut = dictionary['num_cut']
            self.offset_dim = dictionary['offset_dim']
            self.thresholds = dictionary['thresholds']
            self.division = dictionary['division']
            return path
        except:
            print 'Unable to load Descriptor Parameters'
    
    def saveFeatureDescriptorParamToFile(self, outPath):
        
        print 'Saving Descriptor Parameters...'   
        filePath = os.path.join(outPath, 'descriptorParam')
        np.savez(filePath,
                 num_cut = self.num_cut,
                 offset_dim = self.offset_dim,
                 thresholds = self.thresholds,
                 division = self.division
                 )
         
        print 'Dictionary saved in', filePath, '\n'
        return filePath
    
    def getFeatureByFireLaneMap(self, map): 
        
        division = self.division

        feature = np.zeros((1, 2 + (division[0] * division[1])))
#         laneDensityMap = np.zeros(map.shape)
#         laneDensityVector = np.zeros((20, 1000))
        
        mapDim = map.shape
        
        feature[0, 0] = float(mapDim[0]) / self.offset_dim[0]
        feature[0, 1] = float(mapDim[1]) / self.offset_dim[1]

        if mapDim[0] >= division[0] and mapDim[1] >= division[1]:
            
            step_y = (float(mapDim[0]) / division[0])
            step_x = (float(mapDim[1]) / division[1])
            
            index = 2
            for i in range(0, division[0]):
                
                start_y = int(round(i * step_y))
                if (i == (division[0] - 1) or (i+1)*step_y >= mapDim[0]):
                    end_y = mapDim[0]
                else:
                    end_y = int(round((i+1) * step_y))
                
                for j in range(0, division[1]):
                    
                    
                    start_x = int(round(j * step_x))
                    if (j == (division[1] - 1) or (j+1)*step_x >= mapDim[1]):
                        end_x = mapDim[1]
                    else:
                        end_x = int(round((j+1) * step_x))
                    

                    subMap = map[start_y:end_y, start_x:end_x]
                    subDim = subMap.shape
                    lanePer = float((subMap > 0).sum()) / (subDim[0] * subDim[1])
                    feature[0, index] = lanePer
                    
#                     laneDensityMap[start_y:end_y, start_x:end_x] = lanePer
#                     laneDensityVector[:, 10*(index-2):10*(index-1)] = lanePer
                    index += 1
                    
                    
                    
#         laneDensityMap = laneDensityMap *255
#         laneDensityVector = laneDensityVector * 255
#         cv.imwrite('/Users/sephon/Desktop/Research/VizioMetrics/image_5752_firelanedensity_vector.jpg', laneDensityVector)
#         plt.imshow(laneDensityVector, cmap = 'gray', interpolation = 'bicubic')
#         plt.show()
                    
#                     plt.imshow(subMap, cmap = 'gray', interpolation = 'bicubic')
#                     plt.show()
        
        return feature
        
    
    def getFeatureByFireLane(self, img):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        imDim = img.shape
        feature = np.zeros((1, 2 + self.num_cut*2))
        feature[0, 0] = float(imDim[0]) / self.offset_dim[0]
        feature[0, 1] = float(imDim[1]) / self.offset_dim[1]
        feature[0, 2:] = SubImageClassifier.getAllDivBlankLine(img, self.thresholds, self.num_cut)
        
        return feature
        
        # Return classification
    def getClassAndProabability(self, img):
        
        X = self.getFeatureByFireLaneMap(img)
        y_pred, y_proba = self.Classifier.predict(X)
        return y_pred, y_proba
    
    def extractCompositeImageFeaure(self, img):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        imDim = img.shape
        feature = np.zeros((1, 2 + self.num_cut*2))
        feature[0, 0] = float(imDim[0]) / self.offset_dim[0]
        feature[0, 1] = float(imDim[1]) / self.offset_dim[1]
        feature[0, 2:] = SubImageClassifier.getAllDivBlankLine(img, self.thresholds, self.num_cut)
        
        return feature

    @ staticmethod
    def whiten(X, M, P):
        X = np.dot((X - M), P)
    


class SubImageClassifier(SVMClassifier):
    
    @ staticmethod
    def indices(a, func):
        return [i for (i, val) in enumerate(a) if func(val)]
    
    @ staticmethod
    def getBlankLine(img, orientation, thresholds): 
        arraySum = np.sum(img, axis = orientation)
        arraySum_nor = arraySum/float(np.max(arraySum))
        arrayVar = np.var(img, axis = orientation) 

        blank_line = SubImageClassifier.indices(zip(arrayVar, arraySum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
        return blank_line
    
    @ staticmethod
    def getBlankArea(img, thresholds):
    
        height, width = img.shape
                
        blank_row = SubImageClassifier.getBlankLine(img, 1, thresholds)
        blank_col = SubImageClassifier.getBlankLine(img, 0, thresholds) 
        
        dimR = len(blank_row)
        dimC = len(blank_col)
        
        return (dimR * width) + dimC * height - dimR * dimC;
    
    # return segmental blank coverage
    @ staticmethod
    def getDivBlankLine(sum_nor, arrayVar, length, thresholds, num_cut):
        
        step = math.ceil(float(length)/num_cut) ##
#         step = (float(length)/num_cut)
#         print step
        result = np.zeros([1, num_cut])

        for i in range(0, num_cut):
            if i*step >= length:
                blank_line = None
            else:
                if (i == (num_cut - 1) or (i+1)*step >= length):
#                     print 'i=', i, int(round(i*step)), length
                    thisSum_nor = sum_nor[int(round(i*step)) : length]
                    thisArrayVar = arrayVar[int(round(i*step)) : length]
    
                else:
#                     print 'i=', i, int(round(i*step)),  int(round((i+1)*step))
                    thisSum_nor = sum_nor[int(round(i*step)) : int(round((i+1)*step))]
                    thisArrayVar = arrayVar[int(round(i*step)) : int(round((i+1)*step))]
                     
    #             print "sum_nor = ", thisSum_nor.shape
                # find the blank row/column by double thresholding
                blank_line = SubImageClassifier.indices(zip(thisArrayVar, thisSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
                # change to opposite
            
            if blank_line is None:
                result[0, i] = 1
            elif len(blank_line) == 0:
                result[0, i] = 0
            else:
                result[0, i] = len(blank_line)/float(max(thisSum_nor.shape))  
         
        return result
    
    @ staticmethod
    def getAllDivBlankLine(img, thresholds, num_cut, debug = False):
    
        height, width = img.shape
        
        rowSum = np.sum(img, axis = 1) #len(rowSum) = height
        colSum = np.sum(img, axis = 0)
        
        rowSum_nor = rowSum/float(np.max(rowSum))
        colSum_nor = colSum/float(np.max(colSum))
        
        rowArrayVar = np.var(img, axis = 1)
        colArrayVar = np.var(img, axis = 0)
        
        result = np.hstack([SubImageClassifier.getDivBlankLine(rowSum_nor, rowArrayVar, height, thresholds, num_cut), SubImageClassifier.getDivBlankLine(colSum_nor, colArrayVar, width, thresholds, num_cut)])
    
        #Debug mode
        if debug:
            blank_row = SubImageClassifier.indices(zip(rowArrayVar,rowSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
            blank_col = SubImageClassifier.indices(zip(colArrayVar,colSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
            SubImageClassifier.showBlankArea(img, blank_row, blank_col, 200, num_cut)
        return result
    
    @ staticmethod
    def getImageFeature(originalImageInfo, img, thresholds, num_cut = 5):
        
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
           
        size = img.shape
        area_per = float(size[0]*size[1])/originalImageInfo['area']
        height_per = float(size[0])/originalImageInfo['height']
        width_per = float(size[1])/originalImageInfo['width']
#         aspect_ratio = float(min(size[0],size[1]))/max(size[1],size[0])
        aspect_ratio_1 = float(size[0])/size[1]
        aspect_ratio_2 = float(size[1])/size[0]
        blank_area_per = SubImageClassifier.getBlankArea(img, thresholds)/float(size[0]*size[1])
        segmental_blank_coverage = SubImageClassifier.getAllDivBlankLine(img, thresholds, num_cut)
        
        feature_vector = np.hstack([np.asmatrix([area_per, height_per, width_per, aspect_ratio_1, aspect_ratio_2, blank_area_per]), segmental_blank_coverage])
        return feature_vector
    
    @ staticmethod
    # debug method
    def showBlankArea(img, blank_row, blank_col, pixel_value, num_cut = 5, title = "unknown"):
        img = np.copy(img)
        img[blank_row,:] = pixel_value
        img[:, blank_col] = pixel_value
        
        plt.title(title)
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        div_x = img.shape[1]/num_cut
        div_y = img.shape[0]/num_cut
        if div_x != 0 and div_y != 0:
            plt.xticks(np.arange(0, img.shape[1]+1, div_x))
            plt.yticks(np.arange(0, img.shape[0]+1, div_y))
        plt.show()
        return img
    
    
    @ staticmethod
    def getImageInfo(img):
        info = {}
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        size = img.shape
        info['height'] = size[0]
        info['width'] = size[1]
        info['area'] = size[0]*size[1]
            
        return info


class Dismantler:
    
    Opt = None
    Classifier = None
    isPreClassified = False
    isMakeTrainingImage = False
    
    def __init__(self, Opt, auxClfPath = None):
        self.Opt = Opt
        self.isPreClassified = Opt.isPreClassified
        self.loadSVMClassifier(auxClfPath)
            
    def loadSVMClassifier(self, auxClfPath):
        try:
            if auxClfPath is not None:
                self.Classifier = SubImageClassifier(self.Opt, isTrain= False, clfPath = auxClfPath)
            else:
                self.Classifier = SubImageClassifier(self.Opt, isTrain= False, clfPath = self.Opt.svmModelPath)
            print 'SubImage Classifier ready. \n'
        except:
            print 'Require valid sub-image classifier. \n'          
        
    # Return classification
    def getSubImageClass(self, subImg, original_image_info):
        thresholds = self.Opt.thresholds
        X = SubImageClassifier.getImageFeature(original_image_info, subImg, thresholds, num_cut = 5)
        y_pred, y_proba = self.Classifier.predict(X)
        return y_pred   
    
    
    def makeTrainData(self, img, save_path, filename, pre_classified = False):
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        self.isMakeTrainingImage = True
        self.current_save_path = save_path
        self.current_filename = filename
        self.pre_classified_for_train = pre_classified
        
        orientation = randint(0,1)
        self.split(img, orientation)
#         self.split(img, 1)
    
    def getSalientMap(self, img, root):
        
        map = np.zeros(img.shape[:2])
        originalImageInfo = SubImageClassifier.getImageInfo(img)
        
        for node in root.traverse():
            if(hasattr(node, 'info')):
                
                subImg = self.getSubImageFromNode(img, node)
                classname = self.getSubImageClass(subImg, originalImageInfo)[0]
                if classname == 'auxiliary':
                    start = node.info['start']
                    end = node.info['end']
                    map[start[0] : end[0], start[1] : end[1]] += np.ones((end[0]-start[0], end[1]-start[1]))
                
        normal = float(np.max(map))
        if normal != 0:
            map = map / float(np.max(map)) * 255
        
        map.astype(int)
        return map
    
    # orientation = 0: vertical firelane
    # orientation = 1: horizontal firelane
    def split(self, img, orientation):       
        
        level = 0
        offset = [0, 0]
        img_id = ""
        original_image_info = SubImageClassifier.getImageInfo(img)
        fire_lane_map = np.zeros(img.shape[0:2])
        root, fire_lane_map = self.split_help(img, original_image_info, Tree(), orientation, level, offset, img_id, fire_lane_map)
        count_standalone = 0
        if self.isPreClassified:
            for node in root.get_leaves():
                if(hasattr(node, 'info')):
                    if node.info['class'] == 'standalone':
                        count_standalone += 1
        
        return root, fire_lane_map, count_standalone      

    def split_help(self, img, original_image_info, root, orientation, level, offset, img_id, fire_lane_map):
        
        thresholds = self.Opt.thresholds
        
        # Pre-processing
        imgDim = img.shape
        arrayDim = imgDim[orientation - 1]        
        blank_line = SubImageClassifier.getBlankLine(img, orientation, thresholds)
        
        # Alternate orientation
        if orientation == 1:
            if len(blank_line) > 0:
                blank_line_loc = np.asarray(blank_line) + offset[0]
                fire_lane_map[blank_line_loc, offset[1]:offset[1]+imgDim[1]] = 1
            next_orientation = 0
#             self.showImg(fire_lane_map)
#             SubImageClassifier.showBlankArea(img, blank_line, [], 122, title = img_id) #debug
        else:
            if len(blank_line) > 0:
                blank_line_loc = np.asarray(blank_line) + offset[1]
                fire_lane_map[offset[0]:offset[0]+imgDim[0], blank_line_loc] = 1
            next_orientation = 1
#             self.showImg(fire_lane_map)
#             SubImageClassifier.showBlankArea(img, [], blank_line, 122, title = img_id) #debug
        
        # find firelanes
        firelane_info = []
        firelane_num = len(blank_line)
        if len(blank_line) > 0:
            firelane_width = 0
            # add head if first column/row is not firelane
            isFireLaneFromHead = True
            if blank_line[0] != 0:
                firelane_info = [[0, 0]] # add head
                isFireLaneFromHead = False
            
            for i, id_line in enumerate(blank_line):      
                firelane_width = firelane_width + 1    

                # firelane
                if i == firelane_num - 1 or blank_line[i+1] - blank_line[i] > 1:   
                    firelane_info.append([id_line, firelane_width])
                    firelane_width = 0

            # Add bottom if last column/row is not firelane
            if blank_line[-1] == arrayDim - 1:
                firelane_info[-1] = [arrayDim, 0]
            else: 
                firelane_info.append([arrayDim, 0]) 
        
            # Remove duplicate head if first firelane starts from the head
            if isFireLaneFromHead:
                firelane_info[0] = [0, 0]
        
                
        if len(firelane_info) > 2:
            for i in range(1, len(firelane_info)):
                this_subImg = TreeNode(name = img_id)
                this_subImg.info = {}
                this_subImg.info['level'] = level 

                if len(firelane_info) > 2: # splits found
                    this_subImg.name = this_subImg.name + str(i-1) + "_"
                    this_subImg.info['level'] = level + 1
                
                this_subImg.info['id'] = this_subImg.name
                
                start = firelane_info[i-1][0]
                end = firelane_info[i][0]
                
                if orientation == 1: #horizontal 
                    subImg = img[start:end, :]       
                    this_subImg.info['start'] = [offset[0] + start, offset[1]] ##
                    this_subImg.info['end'] = [offset[0] + end, offset[1] + imgDim[orientation]] ##
                else: # vertical
                    subImg = img[:, start:end]
                    this_subImg.info['start'] = [offset[0], offset[1] + start] ##
                    this_subImg.info['end'] = [offset[0] + imgDim[orientation], offset[1] + end] ##
                
                this_subImg.info['dim'] = subImg.shape
                this_subImg.info['size'] = this_subImg.info['dim'][0] * this_subImg.info['dim'][1]
                this_subImg.info['blank_area'] = SubImageClassifier.getBlankArea(subImg, thresholds)
                this_subImg.info['aspect_ratio'] = float(min(this_subImg.info['dim']))/max(this_subImg.info['dim'])
                
                default_type = 'unknown'
                this_subImg.info['class'] = default_type
                # Pre-Classified
                if self.isPreClassified:
                    this_subImg.info['class'] = self.getSubImageClass(subImg, original_image_info)[0]
                # For making training image 
                if self.isMakeTrainingImage:
                    self.saveTrainingImage(subImg, original_image_info, this_subImg.name, self.current_save_path, self.current_filename)
                    
                
                # Initial value to next recursion
                next_offset = this_subImg.info['start']
                next_img_id = this_subImg.name
                next_level = level + 1
                
                # Determine recursion need
                isHasSplit = len(firelane_info) > 2
                isLevelOne = (level == 1)
                isStandAlone = this_subImg.info['class'] == 'standalone' or this_subImg.info['class'] == default_type
                
                # Add this subImage as a child
                root.add_child(child = this_subImg, name = this_subImg.name)
                # Keep traversing 
                if (isLevelOne or isHasSplit) and isStandAlone:
                    this_subImg = self.split_help(subImg, original_image_info, this_subImg, next_orientation, next_level, next_offset, next_img_id, fire_lane_map)
   
        # No splitting found in this image            
        if level == 0 and len(root.get_children()) == 0:
                
            this_subImg = TreeNode(name = 0)
            this_subImg.info = {}
            this_subImg.info['id'] = 0
            this_subImg.info['level'] = level + 1
            this_subImg.info['start'] = [0, 0]
            this_subImg.info['end'] = [imgDim[0] , imgDim[1]]
            this_subImg.info['class'] = 'standalone'
            root.add_child(child = this_subImg, name = this_subImg.name)
                 
        return root, fire_lane_map
    
    
    def getEffectiveRegionMask(self, img):
        
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            
        first_vertical, fire_lane_map_vertical, count_standalone_first_vertical = self.split(img, 0)
        first_horizontal, fire_lane_map_horizontal, count_standalone_first_horizontal = self.split(img, 1)
        mask = fire_lane_map_vertical + fire_lane_map_horizontal
        mask = np.divide(mask, np.max(mask)) * 255
        return mask
            
    # return a list of node showing the final segmentation
    def dismantle(self, img):
        
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        
        first_vertical, fire_lane_map, count_standalone_first_vertical = self.split(img, 0)
        first_horizontal, fire_lane_map, count_standalone_first_horizontal = self.split(img, 1)
        
        first_vertical = self.merge(img, first_vertical)
        first_horizontal = self.merge(img, first_horizontal)
    
        final_result = self.select(img, first_vertical, first_horizontal)
        
        return final_result
    
    def merge(self, img, root):
#         print 'show split result' # DEBUG
#         self.showSegmentation(img, root, show = True) # DEBUG
        
        # Check no fragment after splitting.
        # If no fragment, do not merge.
            
        if len(root.get_children()) == 1:
            node_list = root
        else:
            root = self.HeirachicalMerge(img, root)
#             print 'show hMerge result' #DEBUG
#             self.showSegmentation(img, root, show = True) # DEBUG
            node_list = self.TMerge(img, root)
#             print 'show TMerge result' # DEBUG
#             self.showSegmentationByList(img, node_list, show = True) # DEBUG
        return node_list
    
    def HeirachicalMerge(self, img, root):
        
        original_image_info = SubImageClassifier.getImageInfo(img)
        if len(root.get_children()) == 0:
            total_level = 0
        else:
            total_level = 10 # need to pass while only

        while total_level > 1:
            # Get leaves in the deepest level
            total_level = 0
            level_list = []
            for node in root.traverse():
                if(hasattr(node, 'info')):
                    this_level = node.info['level']
                    if this_level > total_level:
                        total_level += 1
                        level_list.append([node])
                    else:
                        level_list[this_level-1].append(node)
            
            this_node_list = level_list[-1]
            
            # Collect parents
            parents = Set([]) # all parents for this node list
            if total_level > 1:
                for node in this_node_list:
                    parent = node.up
                    parents.add(parent)

            # Perform local merge
            if len(this_node_list) > 0:   
                new_node_list = self.localMerge(img, original_image_info, this_node_list)
            
            # Update the tree 
            if total_level > 1: # total_level == 1 : level 1 has been traversed and merged 
                # move newly merged nodes to their parents' level
                for node in new_node_list:
                    # Update node.info
                    id_temp = node.info['id'].split('_')
                    new_id = ''
                    for n in id_temp[:-2]:
                        n += '_'
                        new_id += n 
                    node.info['id'] = new_id
                    node.name = new_id
                    node.info['level'] -= 1
                    # Move up
                    parent = node.up
                    parent.add_sister(node.detach())
                
                # remove old branches (old parents and old children)
                for parent in parents:
                    parent.detach()
            else:
                root = Tree()
                for node in new_node_list:
                    root.add_child(child = node, name = node.info['id'])
            
        return root ###
            
    
    def localMerge(self, img, original_image_info, node_list):
        
        auxiliary_index_list = []
        for i, node in enumerate(node_list):
            this_subImg = self.getSubImageFromImage(img, node.info['start'], node.info['end'])
            this_type = self.getSubImageClass(this_subImg, original_image_info)[0]
            node.info['class'] = this_type
            node.info['target'] = None
            node.info['merged'] = False
            node.info['enlarged'] = False
            node.info['num_as_target'] = 0
            if node.info['class'] == 'auxiliary':###
                auxiliary_index_list.append(i)###
            
            
            
        # Get distance matrix
        dist_matrix = self.getDistMatrix2(img, node_list, self.Opt.thresholds)
        # Get need-merged nodes
        activate_index_list = []
        exceptio_index_list = []
        for i in auxiliary_index_list:
            node = node_list[i]
            target_node_index = int(np.argmin(abs(dist_matrix[i, :])))
            target_node = node_list[target_node_index]
            
            # only small to large is allowed
            if np.inf != dist_matrix[i, target_node_index] >= 0 \
                and target_node.info['target'] != i:
                node.info['target'] = target_node_index
                activate_index_list.append(i)
            elif dist_matrix[i, target_node_index] < 0 and self.isAlinged(node, target_node): #big to small and aligned
                node.info['target'] = target_node_index
                exceptio_index_list.append(i)
               
        # For the case of big to small but the small is not to the big. 
        for i in exceptio_index_list:
            node = node_list[i]
            target_node = node_list[node.info['target']]
            if target_node.info['target'] is None:
                activate_index_list.append(i)
          
        
        # Merge nodes if found
        if len(activate_index_list) > 0:
            for i in activate_index_list:
                
                node = node_list[i]
                node.info['merged'] = True
                target = node.info['target']
                
                # Find final merge destination
                while(node_list[target].info['merged']): #if the target has been merged, it must have a target
                    target = node_list[target].info['target']
                    node.info['target'] = target
                
                # Update Node
                self.mergeNodes(node, node_list[target])
                node_list[target].info['enlarged'] == True
              
            # Remove merged node
            for index in sorted(activate_index_list, reverse=True):
                node = node_list[index]
                del node_list[index]
            
            return self.localMerge(img, original_image_info, node_list)
        else:
            return node_list
        
    # Return a node list of decomposition
    def TMerge(self, img, root):
        
        node_list = []
        for node in root.traverse():
            if(hasattr(node, 'info')):
                node_list.append(node)
#                 print node.info

        # Get distance matrix
        dist_matrix = self.getTMergingDistMatrix(img, node_list, self.Opt.thresholds)
        removed_indice = []
        
#         print node_list
        # Get merging target 
        
#         self.showSegmentationByList(img, node_list, show = True) #DEBUG
        
        for i, node in enumerate(node_list):
            target_node_index = int(np.argmin(abs(dist_matrix[i, :])))
            
            # 1. This node is auxiliary or false_standalone
            # 2. Distance != infinity
            if node.info['class'] != 'standalone' and dist_matrix[i, target_node_index] != np.inf:
                
                
                
                extend_orientation = self.getExtendOrientation(node, node_list[target_node_index])
                target_indice = self.getOverLapNodeindice(i, target_node_index, node_list)
                target_indice.append(target_node_index) # put self into extend list
        
#                 print 'show this node'
#                 self.showSegmentationByList(img, node, show = True) # DEBUG
                
#                 print 'show target node'
#                 self.showSegmentationByList(img, node_list[target_node_index], show = True) # DEBUG
                
#                 print 'target_indice', target_indice
#                 targetNodeList = []
#                 for index in target_indice:
#                     targetNodeList.append(node_list[index])
#                     
#                 print 'targetNodeList', targetNodeList
#                 print 'show involved node'
#                 self.showSegmentationByList(img, targetNodeList, show = True) # DEBUG
                
                
                for index in target_indice:
                    beneficiary_node = node_list[index]
                    if extend_orientation <= 2:  # horizontal
                        beneficiary_node.info['start'][extend_orientation - 1]  = node.info['start'][extend_orientation - 1]
                    else: # vertical
                        beneficiary_node.info['end'][extend_orientation - 3]  = node.info['end'][extend_orientation - 3]
                        
                    new_dim = ((beneficiary_node.info['end'][0] - beneficiary_node.info['start'][0]), \
                                                (beneficiary_node.info['end'][1] - beneficiary_node.info['start'][1]))
                    beneficiary_node.info['dim'] = new_dim
                    beneficiary_node.info['size'] = new_dim[0] * new_dim[1]
                    
                    #ensure the extended nodes' targets won't be the the same nodes that have been traverse
                    dist_matrix[index, i] = np.inf
                        
                removed_indice.append(i)
                
                
                
                
        # Remove merged node
        for index in sorted(removed_indice, reverse=True):
            node = node_list[index]
            node.detach()
            del node_list[index]
            
        return node_list
    
    @ staticmethod
    def select(img, node_list_1, node_list_2):
        
        score_1 = Dismantler.getSelectScore(img, node_list_1, 0.9)
        score_2 = Dismantler.getSelectScore(img, node_list_2, 0.9)
        
        if score_1 > score_2:
            return node_list_1
        else:
            return node_list_2
        
    @ staticmethod
    def getSelectScore(img, node_list, penalty_coefficient):

        if len(node_list) > 1: # no sub-images
            node_list = sorted(node_list, key=lambda node: node.info['size'])
            score = 0
            for node in node_list:
                score += 2 * math.sqrt(node.info['size'])
                
            dist_matrix = Dismantler.getSelectDistMatrix(node_list);
            
            for i in range(0, len(node_list)):
                penalty = int(np.min(dist_matrix[i, :]))
                score -= penalty_coefficient * penalty
                
            score = float(score) / sum(img.shape)
        else:
            score = 1
        return score
        
    @ staticmethod
    def getSelectDistMatrix(node_list):
        
        matrix_dim = len(node_list)
        final_dim = np.ones([len(node_list), 2])
        distance_matrix = np.zeros([matrix_dim, matrix_dim]) + np.inf
        for i, node in enumerate(node_list):
            final_dim[i, :] += node.info['dim']

        for i in range(0, matrix_dim):
            for j in range(i, matrix_dim):
                if i != j:
                    distance = np.sum(abs(final_dim[i, :] - final_dim[j, :]))
                    distance_matrix[i, j] = distance
                    distance_matrix[j, i] = distance
                
        return distance_matrix
        
    
    @ staticmethod
    def mergeNodes(node, target_node):
        
        orientation = Dismantler.getAttachedOrientation(node, target_node)
        if orientation == 'vertical':
            target_node.info['start'][0] = min(target_node.info['start'][0], node.info['start'][0]) 
            target_node.info['end'][0] = max(target_node.info['end'][0], node.info['end'][0])
        elif orientation == 'horizontal':
            target_node.info['start'][1] = min(target_node.info['start'][1], node.info['start'][1]) 
            target_node.info['end'][1] = max(target_node.info['end'][1], node.info['end'][1])
        
        new_dim = (target_node.info['end'][0] - target_node.info['start'][0], \
                                   target_node.info['end'][1] - target_node.info['start'][1])
        
        target_node.info['dim'] = new_dim
        target_node.info['size'] = new_dim[0] * new_dim[1]
        
        return target_node
     
     
    @ staticmethod
    def getTMergingDistMatrix(img, node_list, thresholds):
        dim = len(node_list)
        distance_matrix = np.zeros((dim, dim)) + np.inf 
        
        for i in range (0, dim):
            if node_list[i].info['class'] != 'standalone':
                for j in range(0, dim):
                    if Dismantler.isNeighbor(node_list[i], node_list[j]):
                        this_node = node_list[i]
                        to_node = node_list[j]
                        
                        this_extend_orientation = Dismantler.getExtendOrientation(this_node, to_node)
                        # Get all overlapped nodes
                        target_indice = Dismantler.getOverLapNodeindice(i, j, node_list)
                        
                        isValidCombination = True
                        
                        # Check all overlapped nodes are in the same orientation
                        if len(target_indice) > 0:
                            for n in target_indice:                     
                                if this_extend_orientation != Dismantler.getExtendOrientation(this_node, node_list[n]):
                                    isValidCombination = False
                                    break
                                
                                # Check all overlapped nodes make the same newly merging 
                                new_start_end = [min(this_node.info['start'][0], to_node.info['start'][0]), \
                                                 min(this_node.info['start'][1], to_node.info['start'][1]), \
                                                 max(this_node.info['end'][0], to_node.info['end'][0]), \
                                                 max(this_node.info['end'][1], to_node.info['end'][1])]        
                                
                                itr_new_start_end = [min(this_node.info['start'][0], node_list[n].info['start'][0]), \
                                                     min(this_node.info['start'][1], node_list[n].info['start'][1]), \
                                                     max(this_node.info['end'][0], node_list[n].info['end'][0]), \
                                                     max(this_node.info['end'][1], node_list[n].info['end'][1])]
                                
                                # Check integrity, newly merged sub-images show all have 
                                # the same height(horizontal T merge)/width (vertical T merge)                             
                                if itr_new_start_end[(this_extend_orientation % 2)::2] != new_start_end[(this_extend_orientation % 2)::2]:
                                    isValidCombination = False
                                    break
                        
                        if isValidCombination:
                            distance_matrix[i, j] = Dismantler.getDistance(img, node_list[i], node_list[j], thresholds)
        
        return distance_matrix
                        
                  
    @ staticmethod
    def getOverLapNodeindice(this_node_index, to_node_index, node_list):
        
        this_node = node_list[this_node_index]
        to_node = node_list[to_node_index]
        
        new_start_end = [min(this_node.info['start'][0], to_node.info['start'][0]), \
                     min(this_node.info['start'][1], to_node.info['start'][1]), \
                     max(this_node.info['end'][0], to_node.info['end'][0]), \
                     max(this_node.info['end'][1], to_node.info['end'][1])]
           
        new_contexes = [[new_start_end[0], new_start_end[1]], [new_start_end[2], new_start_end[1]], \
                    [new_start_end[2], new_start_end[3]], [new_start_end[0], new_start_end[3]]]
    
        overlap_node_list = []
        for i, node in enumerate(node_list):
            
            if i != this_node_index and i != to_node_index:
                
                if Dismantler.isNeighbor(node_list[this_node_index], node):
                    node_start_end = node.info['start'] + node.info['end']
                    contexes = [[node_start_end[0], node_start_end[1]], [node_start_end[2], node_start_end[1]], \
                        [node_start_end[2], node_start_end[3]], [node_start_end[0], node_start_end[3]]]
                    
                    for n in range(0,4):
                        if new_contexes[n][0] > contexes[0][0] and new_contexes[n][0] < contexes[2][0] and \
                        new_contexes[n][1] > contexes[0][1] and new_contexes[n][1] < contexes[2][1]:
                            overlap_node_list.append(i)
                            break
                        
                        if contexes[n][0] > new_contexes[0][0] and contexes[n][0] < new_contexes[2][0] and \
                        contexes[n][1] > new_contexes[0][1] and contexes[n][1] < new_contexes[2][1]:
                            overlap_node_list.append(i)
                            break
        
        return overlap_node_list
    
    
    # node_1 | node_2: factor = 2
    # node_2 | node_1: factor = 4
    # node_1
    # ------ = factor = 1
    # node_2
    # node_2
    # ------ = factor = 3
    # node_1    
    @ staticmethod
    def getExtendOrientation(node_1, node_2):
        
        if node_2.info['start'][0] == node_1.info['end'][0]:
            factor = 1
        elif node_1.info['start'][0] == node_2.info['end'][0]:
            factor = 3
        elif node_1.info['end'][1] == node_2.info['start'][1]:
            factor = 2
        elif node_2.info['end'][1] == node_1.info['start'][1]:
            factor = 4
        else:
            factor = None
            
        return factor
    
    
    @ staticmethod
    def getDistMatrix2(img, node_list, thresholds):
        dim = len(node_list)
        distance_matrix = np.zeros((dim, dim)) + np.inf
        for i in range(0, dim):
            if node_list[i].info['class'] == 'auxiliary':
                for j in range(0, dim):
                # determine if the two sub-images are valid for merging
                    if Dismantler.isValidMerging(node_list[i], node_list[j]):
                        distance = Dismantler.getDistance(img, node_list[i], node_list[j], thresholds)
                        if node_list[j].info['class'] == 'standalone':
                            distance_matrix[i, j] = distance
                        if node_list[j].info['size'] >= node_list[i].info['size'] \
                            and Dismantler.isAlinged(node_list[i], node_list[j]):
                            distance_matrix[i, j] = distance
                        else:
                            if distance == 0:
                                distance_matrix[i, j] = -0.5
                            else:
                                distance_matrix[i, j] = -distance
                                
                                
#                         if node_list[j].info['size'] >= node_list[i].info['size']:
#                             if Dismantler.isAlinged(node_list[i], node_list[j]):
#                                 distance_matrix[i, j] = distance  # all good
#                             else:
#                                 distance_matrix[i, j] = -distance # small to big but not aligned
#                         
#                         else:
#                             if distance == 0: # prevent size vague, so that punish big to small with -0.5
#                                 distance_matrix[i, j] = -0.5
#                             elif Dismantler.isAlinged(node_list[i], node_list[j]):
#                                 distance_matrix[i, j] = -distance + 0.25 # big to small but aligned
#                             else:
#                                 distance_matrix[i, j] = -distance # big to small but not aligned
                            
                            
        return distance_matrix
           
    @ staticmethod
    def getDistMatrix(img, node_list, thresholds):
        dim = len(node_list)
        distance_matrix = np.zeros((dim, dim)) + np.inf
        for i in range(0, dim):
            if node_list[i].info['class'] == 'auxiliary':
                for j in range(0, dim):
                # determine if the two sub-images are valid for merging
                    if Dismantler.isValidMerging(node_list[i], node_list[j]):
                        distance = Dismantler.getDistance(img, node_list[i], node_list[j], thresholds)
                        if  node_list[j].info['size'] > node_list[i].info['size']:
                            distance_matrix[i, j] = distance
                        else:
                            distance_matrix[i, j] = -distance
                            
        return distance_matrix
                    
    @ staticmethod
    def getAttachedOrientation(node_1, node_2):
                           
        extend_orientation = Dismantler.getExtendOrientation(node_1, node_2)
        
        if extend_orientation is None:
            return None
        elif extend_orientation % 2 == 1:
            return 'vertical'
        elif extend_orientation % 2 == 0:
            return 'horizontal'
                                             
    @ staticmethod
    def getDistance(img, node_1, node_2, thresholds):
        
        align_orientation = Dismantler.getAttachedOrientation(node_1, node_2)
        
        if align_orientation == 'vertical':
            orientation = 1
                        
        elif align_orientation == 'horizontal':
            orientation = 0
        
        # find pattern head and tail of node_1  
        start_1 = node_1.info['start']
        end_1 = node_1.info['end']
        img_1 = img[start_1[0]:end_1[0], start_1[1]:end_1[1]]  
        non_blank_line_1 = Dismantler.getNonBlankLine(img_1, orientation, thresholds)
        
        head_1 = node_1.info['end'][orientation-1]
        tail_1 = node_1.info['start'][orientation-1]
        if len(non_blank_line_1) > 0:
            head_1 = tail_1 + non_blank_line_1[0]
            tail_1 = tail_1 + non_blank_line_1[-1]
            
        # find pattern head and tail of node_2
        start_2 = node_2.info['start']
        end_2 = node_2.info['end']
        img_2 = img[start_2[0]:end_2[0], start_2[1]:end_2[1]]  
        non_blank_line_2 = Dismantler.getNonBlankLine(img_2, orientation, thresholds)  
        
        head_2 = node_2.info['end'][orientation-1]
        tail_2 = node_2.info['start'][orientation-1]
        if len(non_blank_line_2) > 0:
            head_2 = tail_2 + non_blank_line_2[0]
            tail_2 = tail_2 + non_blank_line_2[-1]
           
        distance = min(abs(head_1 - tail_2), abs(head_2 - tail_1))
        if not Dismantler.isAlinged(node_1, node_2):
            distance += 0.5
        return distance
    
    @ staticmethod
    def getLastFireLaneStart(img, orientation, blank_line):
        previous_line = img.shape[orientation - 1]
        for line in reversed(blank_line):
            if previous_line - line > 1:
                break
            previous_line = line
        return previous_line
                    
    @ staticmethod
    def isSiblings(node_1, node_2, show = False):
        
        #if no fragment from splitting 
#         if isinstance((node_1.info['id']), int):
#             return False
        
        id_1 = node_1.info['id'].split('_')
        id_2 = node_2.info['id'].split('_')
        
        
        id_1[-2] = 'r'
        id_2[-2] = 'r'
        
        if show:
            print "before", node_1.info['id']
            print "then", id_1
            print "before", node_2.info['id']
            print "then", id_2
        
        if id_1 != id_2:
            return False
        
        return True
    
    @ staticmethod
    # 1. Two nodes are siblings
    # 2. Two nodes are attached 
    def isNeighbor(node_1, node_2):
        if not Dismantler.isSiblings(node_1, node_2):
            return False
        
        # Determine attachment
        orientation = Dismantler.getAttachedOrientation(node_1, node_2)
        start_1 = node_1.info['start']
        end_1 =  node_1.info['end']
        dim_1 = node_1.info['dim']
        start_2 = node_2.info['start']
        end_2 =  node_2.info['end']
        dim_2 = node_2.info['dim']  
        center_1 = [float(end_1[0]-start_1[0])/2 + start_1[0], float(end_1[1]-start_1[1])/2 + start_1[1]]
        center_2 = [float(end_2[0]-start_2[0])/2 + start_2[0], float(end_2[1]-start_2[1])/2 + start_2[1]]
        
        if orientation == 'horizontal':
            if abs(center_1[0] - center_2[0]) < float(dim_1[0] + dim_2[0])/ 2 + 1:
                return True
        elif orientation == 'vertical':
            if abs(center_1[1] - center_2[1]) < float(dim_1[1] + dim_2[1])/ 2 + 1:
                return True

        return False

    @ staticmethod
    def getNonBlankLine(img, orientation, thresholds): 
        arraySum = np.sum(img, axis = orientation)
        arraySum_nor = arraySum/float(np.max(arraySum))
        arrayVar = np.var(img, axis = orientation) 
        blank_line = SubImageClassifier.indices(zip(arrayVar, arraySum_nor), lambda x: x[0] >= thresholds['varThres'] and (x[0] >= thresholds['var2Thres'] or x[1] <= thresholds['splitThres']))
        return blank_line  
    
    @staticmethod
    def isAlinged(node, target_node):
        isHorizontalAligned = node.info['start'][0] == target_node.info['start'][0] \
                                and node.info['end'][0] == target_node.info['end'][0]
        
        isVerticalAligned = node.info['start'][1] == target_node.info['start'][1] \
                                and node.info['end'][1] == target_node.info['end'][1]
                                
        return isHorizontalAligned or isVerticalAligned
    
    @staticmethod
    # 1. Two nodes are neighbors
    # 2. Newly merging sub-images does not overlap over sub-images
    def isValidMerging(node, target_node):
        
        # Determine if the target is standalone or larger auxiliary    
        if not Dismantler.isNeighbor(node, target_node):
            return False
        
        return True    
#         isHorizontalAligned = node.info['start'][0] == target_node.info['start'][0] \
#                                 and node.info['end'][0] == target_node.info['end'][0]
#         
#         isVerticalAligned = node.info['start'][1] == target_node.info['start'][1] \
#                                 and node.info['end'][1] == target_node.info['end'][1]
#             
#         return isHorizontalAligned or isVerticalAligned
    

    @ staticmethod
    def showSegmentationByList(img, node_list, show = True):
        
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        for node in node_list:
            if(hasattr(node, 'info')):
                start = node.info['start']
                end = node.info['end']
                plt.plot([start[1], end[1]], [start[0], start[0]],'r')
                plt.plot([end[1], end[1]], [start[0], end[0]],'r')
                plt.plot([start[1], end[1]], [end[0], end[0]], 'r')
                plt.plot([start[1], start[1]], [start[0], end[0]], 'r')        
        
        dim = img.shape
        plt.axis([-10, dim[1] + 10, dim[0] + 10, -10])      
        if show:
            plt.show()
        return plt

    @ staticmethod
    def saveSegmentationLayoutByList(img, node_list, file_path, filename):
        filename = filename.split('.')[0] + '.png'
        saving_path = os.path.join(file_path, filename)
        plt = Dismantler.showSegmentationByList(img, node_list, show = False)
        plt.savefig(saving_path, format='png')
        plt.close()
        print 'Segmentation saved as %s' % saving_path
        

    def saveTrainingImage(self, img, original_image_info, image_name, file_path, filename):
        
        if self.pre_classified_for_train:
            class_name = self.getSubImageClass(img, original_image_info)
        
        tmp = filename.split('.')
        suffix = tmp[1]
        prefix = tmp[0]
        
        new_filename = prefix + '_' + image_name + '.' + suffix

        file_path = os.path.join(file_path, class_name[0])
        save_path = os.path.join(file_path, new_filename)
        print save_path
        Dismantler.saveImage(img, save_path)
    
    @ staticmethod
    def updateImageToEffectiveAreaFromNodeList(img, node_list, thresholds):
        
        removed_indice = []
        for i, node in enumerate(node_list):
            subImg = Dismantler.getSubImageFromNode(img, node)
            heads, ends = Dismantler.getEffectiveImageArea(subImg, thresholds)
        
            start = node.info['start'] 
            end = node.info['end'] 
            node.info['start'] = [start[0] + heads[1], start[1] + heads[0]]
            node.info['end'] = [start[0] + ends[1], start[1] + ends[0]]
            
            # find total blank sub-images
            if (node.info['start'][0] >= node.info['end'][0]) or (node.info['start'][1] >= node.info['end'][1]):
#                 Dismantler.showSegmentationByList(img, node, show = True) # DEBUG
                removed_indice.append(i)
            
        # remove total blank sub-images
        for index in sorted(removed_indice, reverse=True):
            node = node_list[index]
            node.detach()
            del node_list[index]
            
        return node_list
            
    
    @ staticmethod
    def getEffectiveImage(img, thresholds):
        
        heads, ends = Dismantler.getEffectiveImageArea(img, thresholds)
        return img[heads[1]:ends[1], heads[0]:ends[0], :]
    
    @ staticmethod 
    def getEffectiveImageArea(img, thresholds):
        
        img_dim = img.shape
        if len(img.shape) == 3:
            img_mono = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        else:
            img_mono = img
        heads = []
        ends = []
        
        for orientation in range(0, 2):
            blank_line = SubImageClassifier.getBlankLine(img_mono, orientation, thresholds)
            if len(blank_line) > 0:
                if blank_line[0] == 0:
                    head = 0
                    while (head + 1) < len(blank_line):
                        if blank_line[head + 1] - blank_line[head] > 1:
                            break
                        head += 1     
                    heads.append(blank_line[head])
                else:
                    heads.append(0)
                
                if blank_line[-1] == img_dim[(orientation + 1) % 2]-1:
                    end = len(blank_line) - 1
                    while end - 1 >= 0:
                        if blank_line[end] - blank_line[end - 1] > 1:
                            break
                        end -= 1
                    ends.append(blank_line[end])
                else:
                    ends.append(img_dim[(orientation + 1) % 2])
            else:
                heads.append(0)
                ends.append(img_dim[(orientation + 1) % 2])
                
        return heads, ends
    
    @ staticmethod
    def showSegmentation(img, split_tree, show = True):
        
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        for node in split_tree.traverse():
            if(hasattr(node, 'info')):
                start = node.info['start']
                end = node.info['end']
                plt.plot([start[1], end[1]], [start[0], start[0]],'r')
                plt.plot([end[1], end[1]], [start[0], end[0]],'r')
                plt.plot([start[1], end[1]], [end[0], end[0]], 'r')
                plt.plot([start[1], start[1]], [start[0], end[0]], 'r')        
        
        dim = img.shape
        plt.axis([-10, dim[1] + 10, dim[0] + 10, -10])
        if show:     
            plt.show()
            
        return plt
    
    @ staticmethod
    def showImg(img):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
        
    # Return sub-image by given original image and start and end points
    @ staticmethod
    def getSubImageFromImage(img, start, end):
        return img[start[0]:end[0], start[1]:end[1]]
    
    # Return all sub-images by given node list
    @ staticmethod
    def getAllSubImageFromImage(node_list):
        sub_images = []
        for node in node_list:
            sub_images.append(Dismantler.getSubImageFromImage(img, node.info['start'], node.info['end']))
            
        return sub_images
    
    # Return sub-image by given node
    @ staticmethod
    def getSubImageFromNode(img, node):
        return Dismantler.getSubImageFromImage(img, node.info['start'], node.info['end'])
    
    
    @ staticmethod
    def saveImage(img, path):
        cv.imwrite(path, img)
        "Image saved as %s" % path
        
if __name__ == '__main__':
    
    #### Dismantler Test
    
    Opt_Dmtler = Option_Dismantler(isTrain = False)
    Dmtler = Dismantler(Opt_Dmtler)
     
    resultPath = '/Users/sephon/Desktop/Research/VizioMetrics/Dismantler/Result'
#     filename = "/Users/sephon/Desktop/Research/ReVision/code/ImageSeg/corpus/large_corpus/image_368.jpg"
    filename = "/Users/sephon/Desktop/Research/VizioMetrics/test_ori.jpg"
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat0124_pure/visualization/image_1432.jpg"
    img = cv.imread(filename)
#     Dmtler.showImg(img)
  
    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#     root, count_standalone = Dmtler.split(img, 0)
#     Dmtler.showSegmentation(img, root)
#     map = Dmtler.getSalientMap(img, root)
#     Dmtler.showImg(map)
             
    node_list = Dmtler.dismantle(img)
    Dmtler.showSegmentationByList(img, node_list)
    
    #### Composite Figure Detector Test
    
#     startTime = time.time()
#     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Dismantler/train_corpus/ee_cat0124_single_composite/composite/image_5752.jpg"
# #     filename = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Classifier/VizSet_pm_ee_cat0124_pure/visualization/image_1432.jpg"
#     img = cv.imread(filename)
#     if len(img.shape) == 3:
#             img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
#             
#     Opt_Dmtler = Option_Dismantler(isTrain = False)
#     Dmtler = Dismantler(Opt_Dmtler)
#     
#     first_vertical, fire_lane_map_vertical, count_standalone_first_vertical = Dmtler.split(img, 0)
#     first_horizontal, fire_lane_map_horizontal, count_standalone_first_horizontal = Dmtler.split(img, 1)
#     map = fire_lane_map_vertical + fire_lane_map_horizontal
#     map = np.divide(map, np.max(map)) * 255
#     cv.imwrite('/Users/sephon/Desktop/Research/VizioMetrics/image_5752_firelane_map.jpg', map)
#         
#     Dmtler.showImg(map)
#         
#     Opt_CD = Option_CompositeDetector(isTrain = False)
#     CID = CompositeImageDetector(Opt_CD)
# #     map = Dmtler.getEffectiveRegionMask(img)
#     classname, prob = CID.getClassAndProabability(map)
#     print classname
#     print prob
#     print time.time() - startTime

