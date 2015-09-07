import csv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import Levenshtein as L
import re


def name_matching(str1, str2):
    regex = re.compile('[^a-zA-Z]')
    regex.sub(' ', 'ab3d*E') 
    str1_list =  str1.split( )
    str2_list =  str2.split( )
    count = 0
    for letter1 in str1_list:
        for letter2 in str2_list:
            if letter1 == letter2:
                count += 1
                break
            
    return count
        
print name_matching('Clinical & Developmental Immunology', 'Clinical and Developmental Immunology')


file_name_null = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/getNULLJournal.csv'
file_name_source = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/bibliometrics/j2013.tab' 
    
    
source = [a for a,b,c,d in list(csv.reader(open(file_name_source, 'rb'), delimiter='\t'))]
null_data = [a for a,b,c,d in list(csv.reader(open(file_name_null, 'rb'), dialect='excel'))]
num_paper = [d for a,b,c,d in list(csv.reader(open(file_name_null, 'rb'), dialect='excel'))]
# for i, name in enumerate(source):
#     print i, name
      
correspond_candidate = []
L.distance('abc', 'cde')
  
for longname in null_data:
    print "looking for", longname
    same_letter = 0
    distance = 100000
    candidate = []
    for i, source_longname in enumerate(source):
        tmp_l = name_matching(longname, source_longname)
        tmp_d = L.distance(longname, source_longname)
        if tmp_l > same_letter:
            candidate.append([i, source_longname, tmp_l, tmp_d])
            distance = tmp_d
            same_letter = tmp_l
            
        elif tmp_l == same_letter and tmp_d <= distance:
            candidate.append([i, source_longname, tmp_l, tmp_d])
            distance = tmp_d
            same_letter = tmp_l
      
    correspond_candidate.append(candidate)
              
print()
  
for i, candidate in enumerate(correspond_candidate):
    best_candidate = candidate[-1]
    best_match = [best_candidate[1], null_data[i], best_candidate[2]]
    best_match_index = [best_candidate[0], i]
    if  best_candidate[2] <= 20:
        print best_match, num_paper[i]
    
    
            
        




# with open(file_name_source ,'rb') as incsv:
#     reader = csv.reader(incsv, delimiter='\t')
#     reader.next()
#     for row in reader:
#         print row