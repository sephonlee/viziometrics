import sys
sys.path.append("..")

from Classifier.Options import *
from Task.LatestModels import *

import csv
import cv2 as cv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


def saveCSV(path, filename, content = None, header = None, mode = 'wb', consoleOut = True):

    if consoleOut:
        print 'Saving image information...'
    filePath = os.path.join(path, filename) + '.csv'
    with open(filePath, mode) as outcsv:
        writer = csv.writer(outcsv, dialect='excel')
        if header is not None:
            writer.writerow(header)
        if content is not None:
            for c in content:
                writer.writerow(c)
    
    if consoleOut:  
        print filename, 'were saved in', filePath, '\n'

#query: figure_paper_sub_class.sql
# corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/S3Sampling/sampling_as_singleton'
corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/S3Sampling/example_figure'
dataPath = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/forpaper/'

labelled_data = []
count = 0
for dirPath, dirNames, fileNames in os.walk(corpusPath):   
    for f in fileNames:
        fname, suffix = Common.getFileNameAndSuffix(f)
        if suffix in OPT_DMTLER.validImageFormat:
            
            img = cv.imread(os.path.join(dirPath, f))            
            imDim = img.shape
            
            img_id = f;
            pmcid = int(f.split('_')[0][3:])
            img_loc = "pubmed/img/" + img_id
            true_class_name = dirPath.split('/')[-1]
            img_height = imDim[0]
            img_width = imDim[1]
            if(true_class_name == "diagram"):
                true_class_name = 'scheme'
            elif(true_class_name == "multi-diagram"):
                true_class_name = 'multi-scheme'
            labelled_data.append([img_id, pmcid, img_loc, true_class_name, img_width, img_height])
            count += 1
            

print labelled_data
header = ["img_id", "pmcid", "img_loc", "true_class_name", "img_width", "img_height"]
saveCSV(dataPath, "example_figure", content = labelled_data, header = header, mode = 'wb', consoleOut = True)


# corpusPath = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/S3Sampling/sampling_is_composite'
# labelled_data = []
# count = 0
# for dirPath, dirNames, fileNames in os.walk(corpusPath):   
#     for f in fileNames:
#         fname, suffix = Common.getFileNameAndSuffix(f)
#         if suffix in OPT_DMTLER.validImageFormat:
#             print dirPath, f
#             img_id = f;
#             pmcid = int(f.split('_')[0][3:])
#             img_loc = "pubmed/img/" + img_id
#             true_class_name = dirPath.split('/')[-1]
#             labelled_data.append([img_id, pmcid, img_loc, true_class_name])
#             count += 1
#             
# 
# print labelled_data
# saveCSV(dataPath, "example_figure", content = labelled_data, header = None, mode = 'ab', consoleOut = True)


# labels = ['Table Precision', 'Photo Precision', 'Data Visualization Precision', 'Diagram Precision', 'Overall Precision']
# types = ['table', 'photo', 'visualization', 'scheme', 'equation']
# symbols = ['o-', 'o-', 'o-', 'o-', 'd-' ]
# type_colors = ['#DBDB8D', '#7F7F7F', '#C49C94', '#1F77B4', 'black']
# group_sizes = [12, 11, 12, 11, 61]
# fig = plt.figure()
# ax = fig.add_subplot(1,1,1)
# count_null = 0
# count_all = 0
# count_valid = 0
# for t in range(0,5):
#     group_size = group_sizes[t]
#     data = []
#     with open(dataPath ,'rb') as incsv:
#         reader = csv.reader(incsv, dialect='excel')
#         reader.next()
#         for row in reader:
#             count_all += 1
#             if row[1] == 'NULL':
#                 count_null += 1
#              
#             if row[1] != 'NULL' and (str(row[3]) == types[t] or 'full' == types[t]):# and img_trueClass_list[row[0]].split('-')[0] != 'multi':
#                 count_valid += 1
#                 class_name = str(row[3]);
#                 if class_name == 'scheme':
#                     class_name = 'diagram'
#                 tmp = { 'img_id': row[0],
#                        'EF': float(row[1]),
#                        'is_composite': int(row[2]),
#                        'classified_class_name': class_name,
#                        'true_class_name': img_trueClass_list[row[0]]}  
#                  
#                 data.append(tmp)
#      
#          
#     print len(data), types[t]
#     num_valid_paper = len(data)
#     raw_eigen_factor = np.zeros([len(data)])
#     raw_correct = np.zeros([len(data)])
#      
#     tmpEF = 1
#     paper_count = 0
#     all_EF = {}
#     list_paper_count = [] 
#     list_change_ef_index = [0]
#     list_pdf_paper_count = []
#     for i, row in enumerate(data):
#         if (tmpEF - row['EF'] > 0.000000000001):
#             all_EF[tmpEF] = paper_count
#             tmpEF = row['EF']
#             if paper_count > group_size:
#                 list_change_ef_index.append(i)
#                 list_paper_count.append(paper_count)
#                 list_pdf_paper_count.append(float(i)/num_valid_paper)
#                 paper_count = 0
#          
#         paper_count += 1
#         if i == len(data) - 1:
#             list_paper_count.append(paper_count)
#             list_pdf_paper_count.append(float(i)/num_valid_paper)
#          
#         if row['classified_class_name'] == row['true_class_name']:
#             raw_correct[i] = 1
#         raw_eigen_factor[i] = row['EF']
#          
#     list_change_ef_index.append(len(data))
#     num_bin = len(list_change_ef_index) - 1
#     print len(list_paper_count)
#      
#     ## Grouping
#     capacity = float(len(data)) / (num_bin)
#     group_accuracy = np.zeros([num_bin])
#     index = 0
#     for i in range(0, len(list_change_ef_index)-1):
#         start_index = list_change_ef_index[i]
#         end_index = list_change_ef_index[i+1]
#         group_accuracy[index] = np.sum(raw_correct[start_index:end_index])/ float(end_index - start_index)
#         index += 1
#     print group_accuracy
#     print list_paper_count
#     print list_pdf_paper_count
#     (cor_coef, p_value_cor) = stats.spearmanr(range(1,num_bin+1), group_accuracy)
#     print cor_coef, p_value_cor
#     cor_coefs.append(cor_coef)
#     p_values.append(p_value_cor)
#     ax.plot(range(1,num_bin+1), group_accuracy, 'o-', label= labels[t], color = type_colors[t], linewidth=2.0)
#      
#  
# print cor_coefs
# print p_values
#  
# handles, labels = ax.get_legend_handles_labels()
# ax.legend(handles, labels)
# # ax.xaxis.set_ticks(range(1997,2015,1))
# fontSize = 18
# plt.ylabel('Precision', fontsize = fontSize)
# plt.xlabel('Ranked Impact By Eigenfactor', fontsize = fontSize)
# plt.tick_params(axis='both', which='major', labelsize = fontSize - 2)
# plt.ylim(0, 1.1)
# plt.title('Precision versus Ranked Impact', fontsize = fontSize)
# plt.show()