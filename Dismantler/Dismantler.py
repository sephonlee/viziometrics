from Options_Dismantler import *
from Classifier.Models import SVMClassifier

from ete2 import Tree
from ete2 import TreeNode
from matplotlib import pyplot as plt
from sets import Set

import matplotlib.cm as cm
import random
import math
import time
import cv2 as cv
import numpy as np



# class TreeNode():
#     
#     children = None
#     siblings = None
#     
#     def __init(self, content = None, name):
#         children = []


class SubImage(TreeNode):
        
    start = None
    end = None
    size = None
    blank_area = None
    aspect_ratio = None
    dim = None
    id = None
    level = None
    
    def __init__(self, id):
        super(SubImage, self)
        self.id = id
#         self.name = id
        
    def extend_id(self, value):
        self.id = self.id + str(value) + "_"
        
    def add_level(self):
        self.level = self.level + 1
        
    def __str__(self):
        msg = "subimg id:" + str(self.id) + "\t level:" + str(self.level) + "\n" + \
                "start:" + str(self.start) + "\t end:" + str(self.end) + "\n" + \
                "size:" + str(self.size) + "\t aspect ratio:" +  str(self.aspect_ratio) + "\n" + \
                 "blank area:" + str(self.blank_area) + "\t dimension:" + str(self.dim)
        return msg


class SubImageFeatureDescriptor():

    def __init__(self):
        return
    
    
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
    def getBlankArea(img, thresholds, debug = False):
    
        height, width = img.shape
        
#         rowSum = np.sum(img, axis = 1) #len(rowSum) = height
#         colSum = np.sum(img, axis = 0)
#         
#         rowSum_nor = rowSum/float(np.max(rowSum))
#         colSum_nor = colSum/float(np.max(colSum))
#         
#         rowArrayVar = np.var(img, axis = 1)
#         colArrayVar = np.var(img, axis = 0)
#         
#         blank_row = SubImageClassifier.indices(zip(rowArrayVar,rowSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
#         blank_col = SubImageClassifier.indices(zip(colArrayVar,colSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
#         
        
        blank_row = SubImageClassifier.getBlankLine(img, 1, thresholds)
        blank_col = SubImageClassifier.getBlankLine(img, 0, thresholds) 
        
        dimR = len(blank_row)
        dimC = len(blank_col)
        
        if debug:
            SubImageClassifier.showBlankArea(img, blank_row, blank_col, 122)
        
        #     showBlankArea(img, blank_row, blank_col, 122)
        return (dimR * width) + dimC * height - dimR * dimC;
    
    # return segmental blank coverage
    @ staticmethod
    def getDivBlankLine(sum_nor, arrayVar, length, thresholds, num_cut):
        
    #     step = math.ceil(float(length)/num_cut)
        step = float(length)/num_cut
        result = np.zeros([1, num_cut])

    #     print "length = ", length
        for i in range(0, num_cut):
            if i*step >= length:
                blank_line = []
            else:
                if (i == (num_cut - 1) or (i+1)*step >= length):
                    thisSum_nor = sum_nor[int(round(i*step)) : length]
                    thisArrayVar = arrayVar[int(round(i*step)) : length]
    
                else:
                    thisSum_nor = sum_nor[int(round(i*step)) : int(round((i+1)*step))]
                    thisArrayVar = arrayVar[int(round(i*step)) : int(round((i+1)*step))]
                    
    #             print "sum_nor = ", thisSum_nor.shape
                # find the blank row/column by double thresholding
                blank_line = SubImageClassifier.indices(zip(thisArrayVar, thisSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
                # change to opposite
            if len(blank_line) == 0:
                result[0, i] = 0
            else:
                result[0, i] = len(blank_line)/float(max(thisSum_nor.shape))  
        
        return result
    
    @ staticmethod
    def getAllDivBlankLine(img, thresholds, num_cut):
    
        height, width = img.shape
        
        rowSum = np.sum(img, axis = 1) #len(rowSum) = height
        colSum = np.sum(img, axis = 0)
        
        rowSum_nor = rowSum/float(np.max(rowSum))
        colSum_nor = colSum/float(np.max(colSum))
        
        rowArrayVar = np.var(img, axis = 1)
        colArrayVar = np.var(img, axis = 0)
        
        result = np.hstack([SubImageClassifier.getDivBlankLine(rowSum_nor, rowArrayVar, height, thresholds, num_cut), SubImageClassifier.getDivBlankLine(colSum_nor, colArrayVar, width, thresholds, num_cut)])
    
        #Debug mode
    #     blank_row = indices(zip(rowArrayVar,rowSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
    #     blank_col = indices(zip(colArrayVar,colSum_nor), lambda x: x[0] < thresholds['varThres'] or (x[0] < thresholds['var2Thres'] and x[1] > thresholds['splitThres']))
    #     showBlankArea(img, blank_row, blank_col, 122, 5)
    #     np.squeeze(np.asarray(b)).tolist()
        return result
    
    @ staticmethod
    def getImageFeature(originalImageInfo, img, thresholds, num_cut = 5):
        
        if len(img.shape) == 3:
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
           
        size = img.shape
        area_per = float(size[0]*size[1])/originalImageInfo['area']
        height_per = float(size[0])/originalImageInfo['height']
        width_per = float(size[1])/originalImageInfo['width']
        aspect_ratio = float(size[0])/size[1]
        blank_area_per = SubImageClassifier.getBlankArea(img, thresholds)/float(size[0]*size[1])
        segmental_blank_coverage = SubImageClassifier.getAllDivBlankLine(img, thresholds, num_cut)
        
        feature_vector = np.hstack([np.asmatrix([area_per, height_per, width_per, aspect_ratio, blank_area_per]), segmental_blank_coverage])
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
    
    def __init__(self, Opt, isPreClassified = False):
        self.Opt = Opt
        self.isPreClassified = isPreClassified
        self.loadSVMClassifier()
            
    def loadSVMClassifier(self):
        try:
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
    # orientation = 0: vertical firelane
    # orientation = 1: horizontal firelane
    def split(self, img):       
        
        orientation = 1
        level = 0
        offset = [0, 0]
        img_id = ""
        original_image_info = SubImageClassifier.getImageInfo(img)
        
        root = self.split_help(img, original_image_info, Tree(), orientation, level, offset, img_id)

        return root        

    def split_help(self, img, original_image_info, root, orientation, level, offset, img_id):
        
        thresholds = self.Opt.thresholds
        
        # Pre-processing
        imgDim = img.shape
        arrayDim = imgDim[orientation - 1]        
        blank_line = SubImageClassifier.getBlankLine(img, orientation, thresholds)
        
        # Alternate orientation
        if orientation == 1:
            next_orientation = 0
#             SubImageClassifier.showBlankArea(img, blank_line, [], 122, title = img_id)
        else:
            next_orientation = 1
#             SubImageClassifier.showBlankArea(img, [], blank_line, 122, title = img_id)
        
        
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
                    this_subImg.info['class'] = self.getSubImageClass(this_subImg, original_image_info)[0]
                
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
                    this_subImg = self.split_help(subImg, original_image_info, this_subImg, next_orientation, next_level, next_offset, next_img_id)
   
        return root

    
    def merge(self, img, root):
        
        print "start merging this root:"
        print root    
        root = self.HeirachicalMerge(img, root)
#         self.showSegmentation(img, root)
        
        self.TMerge(img, root)

        
        
        return
    
    def HeirachicalMerge(self, img, root):
        
        original_image_info = SubImageClassifier.getImageInfo(img)
        total_level = 10
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
            
            # Local merging
            this_node_list = level_list[-1]            
            new_node_list = self.localMerge(img, original_image_info, this_node_list)
            
            if total_level > 1: # total_level == 1 : level 1 has been traversed and merged 
                parents = Set([])
                # move newly merged nodes to their parents' level
                for node in new_node_list:
                    # Update node.info
                    id = node.info['id']
                    node.info['id'] = id[0:-2]
                    node.name = id[0:-2]
                    node.info['level'] -= 1
                    
                    # Move up
                    parent = node.up
                    parent.add_sister(node.detach())
                    parents.add(parent)
                
                # remove old branches (old parents and old children)
                for parent in parents:
                    parent.detach()
            else:
                root = Tree()
                for node in new_node_list:
                    root.add_child(child = node, name = node.info['id'])
            
        return root ###
            
    # return a list of newly merged nodes
    def localMerge(self, img, original_image_info, node_list):
           
#         auxiliary_list = []
        
        # Classification
        for node in node_list:    
            this_subImg = self.getSubImageFromImage(img, node.info['start'], node.info['end'])
            this_type = self.getSubImageClass(this_subImg, original_image_info)[0]
            node.info['class'] = this_type
            node.info['target'] = None
            node.info['num_as_target'] = 0
#             if this_type == 'auxiliary':
#                 auxiliary_list.append(node)
                # find nearest neighbor
        
        # Get distance matrix
        dist_matrix = self.getDistMatrix(img, node_list, self.Opt.thresholds)

        ### DEBUG ###
#         print dist_matrix
#         self.showSegmentationByList(img, node_list)
        ### DEBUG ###
        
        # Get merging target
        for i, node in enumerate(node_list):
            target_node_index = int(np.argmin(abs(dist_matrix[i, :])))
            # 1. This node is auxiliary
            # 2. Distance != infinity
            # 3. Prevent target back
            # 4. Distance > 0
            # 5. If distance < 0, the merging target has another target other than this node
            if node.info['class'] == 'auxiliary' and \
                dist_matrix[i, target_node_index] < np.inf and node_list[target_node_index].info['target'] != i and \
               (dist_matrix[i, target_node_index] > 0 or int(np.argmin(abs(dist_matrix[target_node_index, :]))) != i):
                node.info['target'] = target_node_index
                node_list[target_node_index].info['num_as_target'] += 1
        
        # Merging
        removed_indexes = []
        isNoMergingActivity = True
        for i, node in enumerate(node_list):
            if node.info['num_as_target'] == 0 and node.info['target'] is not None:
                isNoMergingActivity = False
                self.mergeNodes(node, node_list[node.info['target']])
                removed_indexes.append(i)
        
        # Remove merged node
        for index in sorted(removed_indexes, reverse=True):
            node = node_list[index]
            del node_list[index]
        
        if isNoMergingActivity:
            return node_list
        else:
            return self.localMerge(img, original_image_info, node_list)
    
    
    
    def TMerge(self, img, root):
        
        node_list = []
        for node in root.traverse():
            if(hasattr(node, 'info')):
                print node.info
                node_list.append(node)
        
        # Get distance matrix
        dist_matrix = self.getTMergingDistMatrix(img, node_list, self.Opt.thresholds)
        print dist_matrix
        removed_indexes = []
        # Get merging target
        for i, node in enumerate(node_list):
            target_node_index = int(np.argmin(abs(dist_matrix[i, :])))
            # 1. This node is auxiliary
            # 2. Distance != infinity
            # 3. Prevent target back
            # 4. Distance > 0
            # 5. If distance < 0, the merging target has another target other than this node
            if node.info['class'] == 'auxiliary' and dist_matrix[i, target_node_index]:
                
                extend_orientation = self.getExtendOrientation(node, node_list[target_node_index])
                target_indxexs = self.getOverLapNodeIndexes(i, target_node_index, node_list)
                
                for index in target_indxexs:
                    beneficiary_node = node_list[index]
                    if extend_orientation <= 2:  # horizontal
                        beneficiary_node.info['start'][extend_orientation]  = node.info['start'][extend_orientation]
                    else:
                        beneficiary_node.info['start'][extend_orientation-2]  = node.info['start'][extend_orientation-2]
                        
                removed_indexes.append(i)
                
        # Remove merged node
        for index in sorted(removed_indexes, reverse=True):
            node = node_list[index]
            node.detach()
            del node_list[index]
                   
        self.showSegmentation(img, root) 
        return root
    
    def select(self):
        return 
        
    def dismantle(self, img):
        return
    
    
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
        print "t matrixs"
        
        for i in range (0, dim):
            if node_list[i].info['class'] == 'auxiliary':
                for j in range(0, dim):
                    print "t-matrixs: i=", i, "j=", j
                    if Dismantler.isNeighbor(node_list[i], node_list[j]):
                        print j, 'is neighbor'
                        this_node = node_list[i]
                        to_node = node_list[j]
                        
                        this_extend_orientation = Dismantler.getExtendOrientation(this_node, to_node)
                        print 'extend orientation:', this_extend_orientation
                        # Get all overlapped nodes
                        target_indexes = Dismantler.getOverLapNodeIndexes(i, j, node_list)
                        print 'target_indexes', target_indexes
                        isValidCombination = True
                        
                        # Check all overlapped nodes are in the same orientation
                        if len(target_indexes) > 0:
                            for n in target_indexes:
                                if this_extend_orientation != Dismantler.getExtendOrientation(node_list[n], to_node):
                                    isValidCombination = False
                                    break
                        
                                # Check all overlapped nodes make the same newly merging 
                                new_start_end = [min(this_node.info['start'][0], to_node.info['start'][0]), \
                                                 min(this_node.info['start'][1], to_node.info['start'][1]), \
                                                 max(this_node.info['end'][0], to_node.info['end'][0]), \
                                                 max(this_node.info['end'][1], to_node.info['end'][1])]        
                                
                                itr_new_start_end = [min(node_list[target_indexes].info['start'][0], to_node.info['start'][0]), \
                                                     min(node_list[target_indexes].info['start'][1], to_node.info['start'][1]), \
                                                     max(node_list[target_indexes].info['end'][0], to_node.info['end'][0]), \
                                                     max(node_list[target_indexes].info['end'][1], to_node.info['end'][1])]
                                if itr_new_start_end != new_start_end:
                                    isValidCombination = False
                                    break
                        
  
                        if isValidCombination:
                            distance_matrix[i, j] = Dismantler.getDistance(img, node_list[i], node_list[j], thresholds)
        
        return distance_matrix
                        
                        
                        
                        
                        
    @ staticmethod
    def getOverLapNodeIndexes(this_node_index, to_node_index, node_list):
        
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
                node_start_end = this_node.info['start'] + this_node.info['end']
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
#             
#             
#             
#             itr_new_start_end = [min(node.info['start'][0], to_node.info['start'][0]), \
#                      min(node.info['start'][1], to_node.info['start'][1]), \
#                      max(node.info['end'][0], to_node.info['end'][0]), \
#                      max(node.info['end'][1], to_node.info['end'][1])]
#             
#             if itr_new_start_end == new_start_end:
#                 overlap_node_list.append(i)
#             
#         return overlap_node_list
    
    
    @ staticmethod
    # node_1 | node_2: factor = 2
    # node_2 | node_1: factor = 4
    # node_1
    # ------ = factor = 1
    # node_2
    # node_2
    # ------ = factor = 3
    # node_1    
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
    def getDistMatrix(img, node_list, thresholds):
        
        dim = len(node_list)
        distance_matrix = np.zeros((dim, dim)) + np.inf
        for i in range(0, dim):
            if node_list[i].info['class'] == 'auxiliary':
                for j in range(0, dim):
                # determine if the two sub-images are valid for merging
                    if Dismantler.isValidMerging(node_list[i], node_list[j]):
                        distance = Dismantler.getDistance(img, node_list[i], node_list[j], thresholds)
                        if  node_list[j].info['size'] >= node_list[i].info['size']:
                            distance_matrix[i, j] = distance
                        else:
                            distance_matrix[i, j] = -distance
                            
        return distance_matrix
                    
    @ staticmethod
    def getAttachedOrientation(node_1, node_2):
        
#         isHorizontallyAttached =  node_1.info['end'][1] == node_2.info['start'][1] \
#                                 or node_2.info['end'][1] == node_1.info['start'][1]
#                                                        
#         isVerticallyAttached =  node_1.info['end'][0] == node_2.info['start'][0] \
#                                  or node_2.info['end'][0] == node_1.info['start'][0]
                           
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
    def isSiblings(node_1, node_2):
        if node_1.info['id'][0:-2] != node_2.info['id'][0:-2]:
            return False
        return True
    
    @ staticmethod
    def isNeighbor(node_1, node_2):
        if not Dismantler.isSiblings(node_1, node_2):
            return False
        
        return Dismantler.getAttachedOrientation(node_1, node_2) is not None

    @ staticmethod
    def getNonBlankLine(img, orientation, thresholds): 
        arraySum = np.sum(img, axis = orientation)
        arraySum_nor = arraySum/float(np.max(arraySum))
        arrayVar = np.var(img, axis = orientation) 
        blank_line = SubImageClassifier.indices(zip(arrayVar, arraySum_nor), lambda x: x[0] >= thresholds['varThres'] and (x[0] >= thresholds['var2Thres'] or x[1] <= thresholds['splitThres']))
        return blank_line
      
    
    @staticmethod
    def isValidMerging(node, target_node):
        
        # Determine if the target is standalone or larger auxiliary
#         if target_node.info['class'] == 'auxiliary' and target_node.info['size'] < node.info['size']:
#             return False
        
        if not Dismantler.isNeighbor(node, target_node):
            return False
            
        isHorizontalAligned = node.info['start'][0] == target_node.info['start'][0] \
                                and node.info['end'][0] == target_node.info['end'][0]
        
        isVerticalAligned = node.info['start'][1] == target_node.info['start'][1] \
                                and node.info['end'][1] == target_node.info['end'][1]
        
#         print "hor:", isHorizontalAligned
#         print "ver:", isVerticalAligned
        
        return isHorizontalAligned or isVerticalAligned
    
    @ staticmethod
    def getSubImageFromImage(img, start, end):
        return img[start[0]:end[0], start[1]:end[1]]
    
    @ staticmethod
    def showSegmentationByList(img, node_list):
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
        plt.show()
    
    @ staticmethod
    def showSegmentation(img, split_tree):
        
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
        plt.show()
    
    @ staticmethod
    def showImg(img):
        plt.imshow(img, cmap = 'gray', interpolation = 'bicubic')
        plt.show()
        
if __name__ == '__main__':
    
    Opt_clf = Option_Dismantler(isTrain = False)
    Dmtler = Dismantler(Opt_clf, isPreClassified = False)
    
    filename = "/Users/sephon/Desktop/Research/ReVision/code/ImageSeg/corpus/pm_ee_cat_0_multi/image_380.jpg"
    img = cv.imread(filename)


    if len(img.shape) == 3:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
          
    root = Dmtler.split(img)
    
    nodes = root.search_nodes(name = '0_')
#     print nodes[0].info
#     
#     print nodes[0].up.info
    
#     Dmtler.showSegmentation(img, root)

#     print root
    Dmtler.merge(img, root)
    
#     nodelist = root.get_leaves_by_name('3_4_1_0_')
#     node_1 = nodelist[0]
#     parent = node_1.up
#     parent.remove_child(node_1)
#     print root

    
    
#     node_1.add_sister(name = 'sister of 3_4_1_0_')
#     print root
    
    
#     nodelist = root.get_leaves_by_name('3_4_1_1_')
#     node_2 = nodelist[0]
#     
#     Dmtler.getDistance(img, node_1, node_2, Opt_clf.thresholds)
    

    
#     print ""
#     nodelist = root.get_leaves_by_name('3_2_1_1_0_')
#     sisList = nodelist[0].get_sisters()
#     print nodelist[0].info
#     print nodelist[0].info['id'][0:-2]

#     for node in sisList:
#         print node.info
#         print node.info['id'][0:-2]

#     node_1 = TreeNode(name = 'node_1')
#     node_1.info = {}
#     node_1.info['class'] = 'auxiliary'
#     node_1.info['id'] = '3_2'
#     node_1.info['start'] = [5, 10]
#     node_1.info['end'] = [25, 100]
#     node_1.info['size'] = 100
#      
#     node_2 = TreeNode(name = 'node_2')
#     node_2.info = {}
#     node_2.info['class'] = 'auxiliary'
#     node_2.info['id'] = '3_1'
#     node_2.info['start'] = [25,10]
#     node_2.info['end'] = [80, 100]
#     node_2.info['size'] = 50
#      
#      
#     print Dmtler.isSiblings(node_2, node_1)
#     print Dmtler.isNeighbor(node_2, node_1)
#     print Dmtler.isValidMerging(node_2, node_1)