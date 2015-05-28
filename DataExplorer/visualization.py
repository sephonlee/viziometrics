import csv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt

file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/ef_figure_identification_tumorigenic.csv'
data = []
maxX = 0
maxY = 0

diag_count = 0

with open(file_name ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel')
    reader.next()
    for row in reader:
        if int(row[12]) != 0:
            diag_count += 1
        
        if float(row[6]) != 0:
#         print row
            figure_per_page = 0
            if int(row[7]) != 0:
                figure_per_page = (int(row[11])+int(row[12]))/float(int(row[7]))
            
            if float(row[5]) > 1:
                print row[3]
                print row
                
            diagram_per = 0
            if int(row[6]) != 0:
                diagram_per = float(int(row[10]))/ int(row[6])
                
            
                
            tmp = {'topic': row[0],
                   'cluster': row[1],
                   'longname': row[2],
                   'eigen_factor': float(row[5]),
                   'num_figures': int(row[6]),
                   'num_pages': int(row[7]),
                   'num_equations': int(row[8]),
                   'num_tables': int(row[9]),
                   'num_photos': int(row[10]),
                   'num_visualizations': int(row[11]),
                   'num_diagrams': int(row[12]),
                   'figure_per_page': figure_per_page,
                   'diagram_per_figure': diagram_per}
            data.append(tmp)
    #         print tmp['figure_per_page']
            maxX = max(maxX, tmp['figure_per_page'])
            maxY = max(maxY, tmp['eigen_factor'])
        
print len(data)
print float(diag_count)/len(data)
x = np.zeros([1, len(data)])
y = np.zeros([1, len(data)])
# for i, row in enumerate(data):    
#     x[0, i] = row['figure_per_page']
#     y[0, i] = row['eigen_factor']
#   
# plt.scatter(y, x, alpha=0.5)
# plt.axis([0, maxY * 1.1, 0, maxX * 1.1])
# plt.ylabel('Figure/Page')
# plt.xlabel('Eigenfactor')
# plt.show()
 
 
for i, row in enumerate(data):    
    x[0, i] = row['diagram_per_figure']
    y[0, i] = row['eigen_factor']
     
plt.scatter(y, x, alpha=0.5)
plt.axis([0, maxY * 1.1, 0, 2])
plt.ylabel('Diagram/Figure')
plt.xlabel('Eigenfactor')
plt.show()
#   


    

    
# print 'spearman'
# # print x
# # print y
# print x.flatten().tolist()
# print stats.spearmanr(range(1, len(data)+1), x.flatten())
# rd = np.random.randn(len(data),1).flatten()
# print stats.spearmanr(range(1, len(data)+1), rd)
    
for i, row in enumerate(data):    
    x[0, i] = row['diagram_per_figure']
    y[0, i] = row['eigen_factor']

num_bin = 20
capacity_ = float(len(data)) / (num_bin)
fp = np.zeros([1, num_bin])
ef = np.zeros([1, num_bin])
index = 0

print 'capacity', capacity_
for i in range(1, num_bin + 1):
    start_index = int(math.ceil((i-1) * capacity_))
    end_index = int(math.ceil(i * capacity_))
    fp[0, index] = sum(x[0,start_index:end_index])/(end_index-start_index)
    ef[0, index] = sum(y[0,start_index:end_index])
    
    print 'index,', index
    print 'start: %d, end: %d, len: %d' %(start_index, end_index, x[0,start_index:end_index].shape[0])
    index += 1
    
plt.scatter(ef, fp, alpha=0.5)
plt.axis([0, np.max(ef) * 1.1, 0, np.max(fp) * 1.1])
plt.ylabel('Visualization/Figure')
plt.xlabel('Eigenfactor')
plt.show()

index = range(1, num_bin + 1)
bar_width = 0.35
bar_value = fp.flatten().tolist()
# print len(bar_value)
# print len(index)
rects1 = plt.bar(index, bar_value, bar_width,
                 color='b',
                 label='Men')

print index
print bar_value

# index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# bar_value = [0.4237811392052982, 0.3922307558988821, 0.75385662475395036, 0.45379875469493915, 0.5420367119596763, 0.34405308949463403, 0.4176783913366069, 0.32222143889424404, 0.222551020921256, 0.10886660693238377, 0.77092114150032776, 0.6267594494032039, 0.89721881075761086, 0.3119435966748183, 0.2597394051140754, 0.2890633517708081, 0.2983825077921195, 0.1142325946723258, 0.66516609120692053, 0.4833043199827445]
(cor_coef, p_value) = stats.spearmanr(index, bar_value)
 
# print cor_coef
# print p_value
 
plt.ylabel('Photo/Figure')
plt.xlabel('Ranking')
plt.title('Identification Tumorigenic')
plt.text(num_bin*7/10, max(bar_value), 'correlation coefficient: %f' % cor_coef)
plt.text(num_bin*7/10, max(bar_value) * 19/20, 'p-value: %f' % p_value)
plt.show()


# num_bin = 20
# capacity = len(data) / (num_bin-1)
# count = 0
# acum_dfp = 0
# acum_ef = 0
# fp = np.zeros([1, num_bin])
# ef = np.zeros([1, num_bin])
# capacities = []
# count = 0
# index = 0
# 
# print capacity
# for i, row in enumerate(data):
#     if i != 0 and i % capacity == 0:
#         fp[0, index] = acum_dfp / count
#         ef[0, index] = acum_ef
#         capacities.append(count)
# #         print i, index, count, acum_ef, acum_dfp / capacity
#         index += 1
#         count = 0
#         acum_ef = row['eigen_factor']
#         acum_dfp = row['diagram_per_figure']
#  
#     elif i == len(data) - 1:
#         acum_dfp += row['diagram_per_figure']
#         acum_ef += row['eigen_factor']  
#         fp[0, num_bin-1] = acum_dfp / count
#         ef[0, num_bin-1] = acum_ef
#         capacities.append(count)
#         print i, index, count, acum_ef, acum_dfp / capacity
#         next
#     else:
#         acum_dfp += row['diagram_per_figure']
#         acum_ef += row['eigen_factor']  
#     count += 1  
#         
#     
# plt.scatter(ef, fp, alpha=0.5)
# plt.axis([0, np.max(ef) * 1.1, 0, np.max(fp) * 1.1])
# plt.ylabel('Visualization/Figure')
# plt.xlabel('Eigenfactor')
# plt.show()
#    
# index = range(1, num_bin + 1)
# bar_width = 0.35
# bar_value = fp.flatten().tolist()
# # print len(bar_value)
# # print len(index)
# rects1 = plt.bar(index, bar_value, bar_width,
#                  color='b',
#                  label='Men')
# (cor_coef, p_value) = stats.spearmanr(index, bar_value)
#  
# plt.ylabel('Diagram/Figure')
# plt.xlabel('Ranking')
# plt.title('Protein Database')
# plt.text(num_bin*7/10, max(bar_value), 'correlation coefficient: %f' % cor_coef)
# plt.text(num_bin*7/10, max(bar_value) * 19/20, 'p-value: %f' % p_value)
# plt.show()
#    
# print 'spearman'
# print stats.spearmanr(index, bar_value)
# print stats.spearmanr([1,2,3,4,5],[2,6,4,1,9])













#     if i == 0 or i % capacity != 0:
#         acum_dfp += row['diagram_per_figure']
#         acum_ef += row['eigen_factor']
#     elif i == len(data) - 1:
#         x[0, num_bin] = acum_dfp / capacity
#         y[0, num_bin] = acum_ef/ capacity
#         capacities.append(count)
#     else:
#         x[0, index] = acum_dfp / capacity
#         y[0, index] = acum_ef / capacity
#         capacities.append(count)
#         print i, index, count
#         index += 1
#         acum_ef = 0
#         acum_dfp = 0
#         count = 0
#     
# print capacities
# file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/ef_figure_treatment high.csv'
# data = []
# with open(file_name ,'rb') as incsv:
#     reader = csv.reader(incsv, dialect='excel')
#     reader.next()
#     for row in reader:
#         tmp = {'longname': row[0],
#                'eigen_factor': float(row[3]),
#                'num_figures': int(row[4]),
#                'num_pages': int(row[5]),
#                'num_equations': int(row[6]),
#                'num_tables': int(row[7]),
#                'num_photos': int(row[8]),
#                'num_visualizations': int(row[9]),
#                'num_diagrams': int(row[10]),}
#         data.append(tmp)
#         
# print len(data)
# 
# x = np.zeros([1, len(data)])
# print x.shape
# y = np.zeros([1, len(data)])
# for i, row in enumerate(data):
# #     x[0, i] = float(row['num_figures']) / row['num_papers']
# #     print row['num_figures'], row['num_pages']
#     if row['num_pages'] == 0:
#         x[0, i] = 0
#     else:
#         x[0, i] = (row['num_diagrams'] + row['num_visualizations'])/float(row['num_pages'])
#     y[0, i] = row['eigen_factor']
# 
# print x
# print y
# plt.scatter(y, x, alpha=0.5)
# plt.axis([0, 0.0001, 0, 10])
# plt.show()


   
# file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/paper_ranking_group.csv'
# data = []
# with open(file_name ,'rb') as incsv:
#     reader = csv.reader(incsv, dialect='excel')
#     reader.next()
#     for row in reader:
#         tmp = {'group_id': row[0],
#                'eigen_factor': float(row[1]),
#                'num_figures': int(row[2]),
#                'num_pages': int(row[3]),
#                'num_equations': int(row[4]),
#                'num_tables': int(row[5]),
#                'num_photos': int(row[6]),
#                'num_visualizations': int(row[7]),
#                'num_diagrams': int(row[8]),}
#         data.append(tmp)
#         
# print len(data)
# 
# x = np.zeros([1, len(data)])
# print x.shape
# y = np.zeros([1, len(data)])
# for i, row in enumerate(data):
# #     x[0, i] = float(row['num_figures']) / row['num_papers']
#     x[0, i] = row['num_figures']/float(row['num_pages'])
#     y[0, i] = row['eigen_factor']
# 
# print x
# print y
# print y[y>1]
# plt.scatter(y, x, alpha=0.5)
# plt.axis([0, 0.001, 0, 2.5])
# plt.show()

        
        


# file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/average_ef.csv'
# data = []
# with open(file_name ,'rb') as incsv:
#     reader = csv.reader(incsv, dialect='excel')
#     reader.next()
#     for row in reader:
#         tmp = {'journal_name': row[0],
#                'eigen_factor': float(row[1]),
#                'num_papers': int(row[2]),
#                'num_figures': int(row[3]),
#                'num_equations': int(row[4]),
#                'num_tables': int(row[5]),
#                'num_photos': int(row[6]),
#                'num_visualizations': int(row[7]),
#                'num_diagrams': int(row[8]),
#                'avg_figure': float(row[9]),}
#         data.append(tmp)
#         
# print len(data)
# 
# x = np.zeros([1, len(data)])
# print x.shape
# y = np.zeros([1, len(data)])
# for i, row in enumerate(data):
# #     x[0, i] = float(row['num_figures']) / row['num_papers']
#     x[0, i] = row['num_equations']/float(row['num_figures'])
#     y[0, i] = row['eigen_factor']
# 
# print x
# print y
# plt.scatter(x, y, alpha=0.5)
# plt.axis([0, 1, 0, 0.0000007])
# plt.show()
# 
#         