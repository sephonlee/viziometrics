import csv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

def getRanking(data):
     
    rankings = np.zeros([data.shape[0]])
    ranking = 1
    
    previous_value = data[0]
    
    for i in range(0, data.shape[0]):
        if data[i] != previous_value:
            ranking += 1
            previous_value = data[i]
            
        rankings[i] = ranking
    return rankings


 

###
file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/figure_paper_sub_class_1997-2014.csv'
num_bin = 50
### /var/www/html/DB/figures_paper_composite.sql --> For all papers
### /var/www/html/DB/topic_ef_figure.sql  --> Select Topics
### Line 27,28, Filter Papers
### Line 32, Change numerator
### Line 37, Change numerator
### Line 69-78, Resort if any demand
### Line 118,119, Assign x,y variables to plot scatter
### Line 139,140, Assign x,y variables to plot barchart

data = []
with open(file_name ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel')
    reader.next()
    for row in reader:
        
        if float(row[2]) != 0: # Filter paper with EigenFactor == 0
#         if float(row[4]) != 0: # Filter papers with page == 0
            figure_per_page = 0
            # Count Figure/Page
            if float(row[4]) != 0:
                figure_per_page = (int(row[6]))/float(int(row[4])) ##############
                
            # Count proportional figure
            proportional_figure = 0
            if (int(row[6]) + int(row[7])+ int(row[8])+ int(row[9])) != 0:
                proportional_figure = float(int(row[9]))/ ( int(row[6]) + int(row[7])+ int(row[8])+ int(row[9])) ###########
            
            tmp = { 'longname': row[0],
                   'eigen_factor': float(row[2]),
                   'num_figures': int(row[3]),
                   'num_pages': int(row[4]),
                   'num_equations': int(row[5]),
                   'num_tables': int(row[6]),
                   'num_photos': int(row[7]),
                   'num_visualizations': int(row[8]),
                   'num_diagrams': int(row[9]),
                   'figure_per_page': figure_per_page,
                   'proportional_figure': proportional_figure}  
            
            data.append(tmp)
        
num_valid_paper = len(data)
print 'number of paper: ', num_valid_paper

raw_proportional_figure = np.zeros([len(data)])
raw_eigen_factor = np.zeros([len(data)])
raw_figure_per_page = np.zeros([len(data)])
raw_page = np.zeros([len(data)])
raw_figure = np.zeros([len(data)])

tmpEF = 1
paper_count = 0
all_EF = {}
list_paper_count = [] 
list_change_ef_index = []
 
for i, row in enumerate(data):

    if (tmpEF - row['eigen_factor'] > 0.00000001):
#         print tmpEF
#         print row['eigen_factor']
        all_EF[tmpEF] = paper_count
        tmpEF = row['eigen_factor']
#         print "next group"
        if paper_count > 800:
            list_change_ef_index.append(i)
            list_paper_count.append(paper_count)
            paper_count = 0
    paper_count += 1
    
    raw_proportional_figure[i] = row['proportional_figure']
    raw_eigen_factor[i] = row['eigen_factor']
    raw_figure_per_page[i] = row['figure_per_page']
    raw_page[i] = row['num_pages']
    raw_figure[i] = row['num_figures']

print "gp", len(list_paper_count)
print list_paper_count
print list_change_ef_index
# print all_EF

row_ranking = getRanking(raw_eigen_factor)

######### Re-Sorting
# sort_indices = np.argsort(raw_page, axis=0)  ######### Sort by page (original data sorted by EigenFactor)
# print sort_indices
# # ori_y = y.copy()
# raw_proportional_figure = raw_proportional_figure[sort_indices][::-1]
# raw_eigen_factor = raw_eigen_factor[sort_indices][::-1]
# raw_figure_per_page = raw_figure_per_page[sort_indices][::-1]
# raw_page = raw_page[sort_indices][::-1]
# raw_figure = raw_figure[sort_indices][::-1]
##########
    
print 'Top 10%: ', np.mean(raw_proportional_figure[0:num_valid_paper/10])
print 'Bottom 90%: ', np.mean(raw_proportional_figure[num_valid_paper/10:])
print 'Bottom 10%: ', np.mean(raw_proportional_figure[num_valid_paper * 9/10:])

print 'Top 50%: ',  np.mean(raw_proportional_figure[0:num_valid_paper/2])
print 'Bottom 50%: ',  np.mean(raw_proportional_figure[num_valid_paper/2:])
print 'All: ', np.mean(raw_proportional_figure)






## Grouping
capacity = float(len(data)) / (num_bin)
group_proportional_figure = np.zeros([num_bin])
group_proportional_figure_std = np.zeros([num_bin])
group_fpp = np.zeros([num_bin])
group_eigen_factor = np.zeros([num_bin])
group_page = np.zeros([num_bin])
group_fpp_std = np.zeros([num_bin])
greaterThenMean = np.zeros([num_bin])
group_proportional_figure_mean = np.mean(raw_proportional_figure)

histogram = np.zeros((num_bin, 10))
greaterThenAvg = []
color = []
index = 0
for i in range(1, num_bin + 1):
    start_index = int(math.ceil((i-1) * capacity))
    end_index = int(math.ceil(i * capacity))
    group_proportional_figure[index] = np.mean(raw_proportional_figure[start_index:end_index])
    group_proportional_figure_std[index] = np.std(raw_proportional_figure[start_index:end_index])
    group_fpp[index] = np.mean(raw_figure_per_page[start_index:end_index])
    group_fpp_std[index] = np.std(raw_figure_per_page[start_index:end_index])
    group_eigen_factor[index] = np.mean(raw_eigen_factor[start_index:end_index])
    group_page[index] = np.mean(raw_page[start_index:end_index])
    
    greaterThenMean[index] =  np.sum(raw_proportional_figure[start_index:end_index] > group_proportional_figure_mean) / float(end_index - start_index)
     
    print 'index: %d, start: %d, end: %d, len: %d' %(index, start_index, end_index, raw_proportional_figure[start_index:end_index].shape[0])
    index += 1
    
    histo = np.histogram(raw_proportional_figure[start_index:end_index], bins=np.arange(0,1.1,0.1))
    histogram[i-1, :] = histo[0]
    greaterThenAvg.append(np.sum(histo[0][3:]))

    print i, (i-1)%2+1, (i-1)/2+1
#     ax = myFig1.add_subplot(num_bin/2, 2, i)
#     ax.bar(histo[1][0:-1], histo[0], 0.05,
#                  label='Men')
#     title = 'rank: %d,  start: %d, end: %d' %(i, start_index, end_index)
#     ax.set_ylim([0,14000])
#     ax.set_title(title)



## Assign X, Y value
x_value = group_eigen_factor ##################
y_value = group_fpp ################
y_std = group_proportional_figure_std ###########

## Plot Scatter
slope, intercept, r_value, p_value, std_err = stats.linregress(x_value, y_value)
plt.plot(x_value, x_value*slope+intercept)
plt.scatter(x_value, y_value, alpha=0.5)
plt.axis([0, np.max(x_value) * 1.1, 0, np.max(y_value) * 1.1])
plt.ylabel('Proportion of Figures that are Classified as Diagrams')
plt.xlabel('Sum of EigenFactor')
plt.text(max(x_value)*9/10, max(y_value) * 2 / 12, 'slope: %f' % slope)
plt.text(max(x_value)*9/10, max(y_value) / 12, 'p-value: %f' % p_value)
plt.show()
 
 
## Calculate correlation and p_value
(cor_coef, p_value_cor) = stats.spearmanr(x_value, y_value)

## Plot Bar Chart
## Assign X, Y value
x_value = group_eigen_factor ##################
y_value = group_fpp ################
# y_value -= np.mean(y_value) 
y_std = group_proportional_figure_std ###########

for y_i in y_value:
    if y_i < 0:
        color.append('r')
    else:
        color.append('b')
        
# index = range(1, num_bin + 1)
index = np.linspace(0,100, (num_bin))
bar_width = 0.3

fig = plt.figure()
ax = fig.add_subplot(1,1,1)

rects1 = ax.bar(index, y_value, bar_width,
                 color=color,
#                 yerr = y_std,
                 label='Men')
 
print 'x', index
print 'y', y_value
  
fmt = '%.0f%%' # Format the ticks, e.g. '40%'
xticks = mtick.FormatStrFormatter(fmt)
# plt.ylabel('Number of Paper with Diagram/Figure Greater Then Average')
plt.ylabel('Proportion of Figures that are Classified as Diagrams')
plt.xlabel('Ranking via Average EigenFactor')
plt.title('Papers From PubMed Central')
plt.text(num_bin*8/10, max(y_value) * 19/20, 'correlation coefficient: %f' % cor_coef)
plt.text(num_bin*8/10, max(y_value) * 18/20, 'p-value: %f' % p_value_cor)
# plt.axis([-3, 103, np.min(y_value) * 1.1, np.max(y_value) * 1.1])
ax.xaxis.set_major_formatter(xticks)
plt.show()



#         