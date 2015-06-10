import csv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/figures_paper_composite_filter_copy.csv'
data = []
maxX = 0
maxY = 0

diag_count = 0

with open(file_name ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel')
    reader.next()
    for row in reader:
        if int(row[10]) != 0:
            diag_count += 1
        
        if float(row[4]) != 0:
#         print row
            figure_per_page = 0
            if int(row[4]) != 0:
                figure_per_page = (int(row[6]))/float(int(row[4]))
            
#             if float(row[3]) > 1:
#                 print row[3]
#                 print row
                
            diagram_per = 0
            if (int(row[5]) + int(row[7])+ int(row[8])+ int(row[9])+int(row[10])) != 0:
                diagram_per = float(int(row[5]))/ ( int(row[5]) + int(row[7])+ int(row[8])+ int(row[9])+int(row[10]))
                
            
                
            tmp = { 'longname': row[0],
                   'eigen_factor': float(row[2]),
                   'num_figures': int(row[3]),
                   'num_pages': int(row[4]),
                   'num_composite': int(row[5]),
                   'num_equations': int(row[6]),
                   'num_tables': int(row[7]),
                   'num_photos': int(row[8]),
                   'num_visualizations': int(row[9]),
                   'num_diagrams': int(row[10]),
                   'figure_per_page': figure_per_page,
                   'diagram_per_figure': diagram_per}
            data.append(tmp)
    #         print tmp['figure_per_page']
            maxX = max(maxX, tmp['figure_per_page'])
            maxY = max(maxY, tmp['eigen_factor'])
        
num_valid_paper = len(data)
print 'number of paper', len(data)
# print '', float(diag_count)/len(data)




x = np.zeros([len(data)])
y = np.zeros([len(data)])
z = np.zeros([len(data)])
p = np.zeros([len(data)])
f = np.zeros([len(data)])
 
for i, row in enumerate(data):    
    x[i] = row['diagram_per_figure']
    y[i] = row['eigen_factor']
    z[i] = row['figure_per_page']
    p[i] = row['num_pages']
    f[i] = row['num_figures']
    
print 'Top 10%: ', np.mean(x[0:num_valid_paper/10])
print 'Bottom 90%: ', np.mean(x[num_valid_paper/10:])
print 'Bottom 10%: ', np.mean(x[num_valid_paper * 9/10:])

print 'Top 50%: ',  np.mean(x[0:num_valid_paper/2])
print 'Bottom 50%: ',  np.mean(x[num_valid_paper/2:])
print 'All: ', np.mean(x)


num_bin = 20
capacity = float(len(data)) / (num_bin)
df = np.zeros([num_bin])
df_std = np.zeros([num_bin])
fp = np.zeros([num_bin])
ef = np.zeros([num_bin])
page = np.zeros([num_bin])
fp_std = np.zeros([num_bin])
greaterThenMean = np.zeros([num_bin])
color = []
index = 0

print
print
print 'capacity', capacity

df_mean = np.mean(x)
print df_mean
print np.sum(x>df_mean)

for i in range(1, num_bin + 1):
    start_index = int(math.ceil((i-1) * capacity))
    end_index = int(math.ceil(i * capacity))
    df[index] = np.mean(x[start_index:end_index])
    df_std[index] = np.std(x[start_index:end_index])
    fp[index] = np.mean(z[start_index:end_index])
    fp_std[index] = np.std(z[start_index:end_index])
    ef[index] = np.mean(y[start_index:end_index])
    page[index] = np.mean(p[start_index:end_index])
    
    greaterThenMean[index] =  np.sum(x[start_index:end_index] > df_mean) / float(end_index - start_index)
     
    print 'index,', index
    print 'start: %d, end: %d, len: %d' %(start_index, end_index, x[start_index:end_index].shape[0])
    index += 1

x_value = ef
y_value = df - np.mean(df)
y_std = df_std

for y_i in y_value:
    if y_i < 0:
        color.append('r')
    else:
        color.append('b')
        
print color

slope, intercept, r_value, p_value, std_err = stats.linregress(x_value, y_value)
plt.plot(x_value, x_value*slope+intercept)
plt.scatter(x_value, y_value, alpha=0.5)
plt.axis([0, np.max(x_value) * 1.1, 0, np.max(y_value) * 1.1])
plt.ylabel('Average Table/Figure')
plt.xlabel('Sum of EigenFactor')
plt.text(max(x_value)*9/10, max(y_value) * 2 / 12, 'slope: %f' % slope)
plt.text(max(x_value)*9/10, max(y_value) / 12, 'p-value: %f' % p_value)
plt.show()
 
# index = range(1, num_bin + 1)

index = np.linspace(0,100, (num_bin))

bar_width = 3
bar_value = y_value
bar_std = y_std
# print len(bar_value)
# print len(index)


fig = plt.figure(1, (7,4))
ax = fig.add_subplot(1,1,1)


print len(ef.flatten().tolist())
print len(index)
rects1 = ax.bar(index, bar_value, bar_width,
                 color=color,
#                 yerr = y_std,
                 label='Men')
 
print index
print bar_value
 
# index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# bar_value = [0.4237811392052982, 0.3922307558988821, 0.75385662475395036, 0.45379875469493915, 0.5420367119596763, 0.34405308949463403, 0.4176783913366069, 0.32222143889424404, 0.222551020921256, 0.10886660693238377, 0.77092114150032776, 0.6267594494032039, 0.89721881075761086, 0.3119435966748183, 0.2597394051140754, 0.2890633517708081, 0.2983825077921195, 0.1142325946723258, 0.66516609120692053, 0.4833043199827445]
(cor_coef, p_value_cor) = stats.spearmanr(x_value, bar_value)
# print cor_coef
# print p_value
  
fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
# plt.ylabel('Number of Paper with Diagram/Figure Greater Then Average')
plt.ylabel('Average Equation/Page')
plt.xlabel('Ranking via Average EigenFactor')
plt.title('Papers From Pubmed Central')
plt.text(num_bin*9/10, max(bar_value) * 19/20, 'correlation coefficient: %f' % cor_coef)
plt.text(num_bin*9/10, max(bar_value) * 18/20, 'p-value: %f' % p_value_cor)
ax.xaxis.set_major_formatter(xticks)
plt.show()

##############################################


sort_indices = np.argsort(p, axis=0)  #########
print sort_indices
ori_y = y.copy()
x = x[sort_indices][::-1]
y = y[sort_indices][::-1]
z = z[sort_indices][::-1]
p = p[sort_indices][::-1]
f = f[sort_indices][::-1]
print 'page'
print p

num_bin = 100
capacity = float(len(data)) / (num_bin)
df = np.zeros([num_bin])
df_std = np.zeros([num_bin])
fp = np.zeros([num_bin])
ef = np.zeros([num_bin])
ef_std = np.zeros([num_bin])
greaterThenMean = np.zeros([num_bin])
rank_avg = np.zeros([num_bin])
page = np.zeros([num_bin])
figure = np.zeros([num_bin])
index = 0


print 'capacity', capacity

df_mean = np.mean(x)
print np.sum(x>df_mean)

for i in range(1, num_bin + 1):
    start_index = int(math.ceil((i-1) * capacity))
    end_index = int(math.ceil(i * capacity))
    df[index] = np.mean(x[start_index:end_index])
    df_std[index] = np.std(x[start_index:end_index])
    fp[index] = np.mean(z[start_index:end_index])
    ef[index] = sum(y[start_index:end_index])
    ef_std[index] = np.std(y[start_index:end_index])
    rank_avg[index] = np.mean(sort_indices[start_index:end_index]) + 1
    greaterThenMean[index] =  np.sum(x[start_index:end_index] > df_mean) / float(end_index - start_index)
    page[index] = np.mean(p[start_index:end_index])
    figure[index] = np.mean(f[start_index:end_index])
    
    print 'index,', index
    print 'start: %d, end: %d, len: %d' %(start_index, end_index, x[start_index:end_index].shape[0])
    print 'df', df[index]
    index += 1
     
print 'pages', page
x_value = page
y_value = df
y_std = ef_std


slope, intercept, r_value, p_value, std_err = stats.linregress(x_value, y_value)
plt.plot(x_value, x_value*slope+intercept)
plt.scatter(x_value, y_value, alpha=0.5)
plt.axis([0, np.max(x_value) * 1.1, 0, np.max(y_value) * 1.1])
plt.xlabel('Average Page/Paper ')
plt.ylabel('Average Diagram/Figure')
plt.text(max(x_value)*9/10, max(y_value) * 2 / 12, 'slope: %f' % slope)
plt.text(max(x_value)*9/10, max(y_value) / 12, 'p-value: %f' % p_value)
plt.show()




 
index = range(1, num_bin + 1)
bar_width = 0.35
bar_value = y_value
bar_std = y_std
print len(ef.flatten().tolist())
rects1 = plt.bar(index, bar_value, bar_width,
                 color='r',
#                  yerr = bar_std,
                 label='Men')
 
print index
print bar_value
 
# index = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20]
# bar_value = [0.4237811392052982, 0.3922307558988821, 0.75385662475395036, 0.45379875469493915, 0.5420367119596763, 0.34405308949463403, 0.4176783913366069, 0.32222143889424404, 0.222551020921256, 0.10886660693238377, 0.77092114150032776, 0.6267594494032039, 0.89721881075761086, 0.3119435966748183, 0.2597394051140754, 0.2890633517708081, 0.2983825077921195, 0.1142325946723258, 0.66516609120692053, 0.4833043199827445]
(cor_coef, p_value) = stats.spearmanr(index, bar_value)
  
# print cor_coef
# print p_value
  
plt.ylabel('Average Diagram/Figure')
plt.xlabel('Ranking of Average Page/Paper Ordered in Descending')
plt.title('Papers From Pubmed Central')
plt.text(num_bin*9/10, max(bar_value), 'correlation coefficient: %f' % cor_coef)
plt.text(num_bin*9/10, max(bar_value) * 19/20, 'p-value: %f' % p_value)
plt.show()



x_value = df
y_value = rank_avg
y_std = ef_std

index = range(1, num_bin + 1)
bar_width = 0.35
bar_value = y_value
bar_std = y_std
print len(ef.flatten().tolist())
rects1 = plt.bar(index, bar_value, bar_width,
                 color='r',
#                  yerr = bar_std,
                 label='Men')

(cor_coef, p_value) = stats.spearmanr(index, bar_value)
plt.ylabel('Average Paper Ranking')
plt.xlabel('Ranking of Diagram/Figure Ordered in Descending')
plt.title('Papers From Pubmed Central')
plt.text(num_bin*9/10, max(bar_value), 'correlation coefficient: %f' % cor_coef)
plt.text(num_bin*9/10, max(bar_value) * 19/20, 'p-value: %f' % p_value)
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