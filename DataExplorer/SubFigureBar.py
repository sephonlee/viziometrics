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


def findBoundary(data, index, threshold):
    
    if index == 0:
        return 0

    if index == len(data):
        return index - 1
    
    tmpEF = data[index]['eigen_factor']
    boundary_forward = index
    boundary_backward = index
    boundary = index
    
    for i in range(index + 1, len(data)):
#         print i , (tmpEF - data[i]['eigen_factor'])
        
        boundary_forward = i
        if (tmpEF - data[i]['eigen_factor'] > threshold):
            break;
            
    for i in range(index - 1, 0, -1): 
#         print i , (tmpEF - data[i]['eigen_factor'])
        
        boundary_backward = i
        if (data[i]['eigen_factor'] - tmpEF > threshold): 
            break;
    
    if boundary_forward - index >= index - boundary_backward:
        boundary = boundary_backward
    else:
        boundary = boundary_forward
        
    return boundary
        

###
file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/figure_paper_sub_class_1997-2014_filter_PLos_One.csv'
num_bin = 20
group_size = 1999
group_starts_ends = [[], ]

### /var/www/html/DB/figures_paper_composite.sql --> For all papers
### /var/www/html/DB/topic_ef_figure.sql  --> Select Topics
### Line 27,28, Filter Papers
### Line 32, Change numerator
### Line 37, Change numerator
### Line 69-78, Resort if any demand
### Line 118,119, Assign x,y variables to plot scatter
### Line 139,140, Assign x,y variables to plot barchart

list_startEndPer = [[0, 0.05], [0.05, 0.25], [0.25, 0.5],  [0.5, 1]]
types = ['Table', 'Photo', 'Data Visualization', 'Diagram']
type_colors = ['#DBDB8D', '#7F7F7F', '#C49C94', '#1F77B4']
group_proportional_figure = []
group_fpp = []
intevals = []
group_eigen_factor = []

list_startEnd = []

for type_index in range(6, 10):
    

    data = []
    with open(file_name ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        for row in reader:
            
    #         if float(row[2]) != 0: # Filter paper with EigenFactor == 0
            if float(row[4]) != 0: # Filter papers with page == 0
                figure_per_page = 0
                # Count Figure/Page
                if float(row[4]) != 0:
                    figure_per_page = (int(row[type_index]))/float(int(row[4])) ##############
                    
                # Count proportional figure
                proportional_figure = 0
                if (int(row[6]) + int(row[7])+ int(row[8])+ int(row[9])) != 0:
                    proportional_figure = float(int(row[type_index]))/ ( int(row[6]) + int(row[7])+ int(row[8])+ int(row[9])) ###########
                
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
    # group_size = num_valid_paper/400
    print 'number of paper: ', num_valid_paper
    print 'group_size:', group_size
 
    raw_proportional_figure = np.zeros([len(data)])
    raw_eigen_factor = np.zeros([len(data)])
    raw_figure_per_page = np.zeros([len(data)])
    raw_page = np.zeros([len(data)])
    raw_figure = np.zeros([len(data)])
        
    for i, row in enumerate(data):
        raw_proportional_figure[i] = row['proportional_figure']
        raw_eigen_factor[i] = row['eigen_factor']
        raw_figure_per_page[i] = row['figure_per_page']
        raw_page[i] = row['num_pages']
        raw_figure[i] = row['num_figures']
    
    
    if len(list_startEnd) == 0:
        for se in list_startEndPer:
            list_startEnd.append([findBoundary(data, int(num_valid_paper * se[0]), 0.000000000001), findBoundary(data, int(num_valid_paper * se[1]), 0.000000000001)])
    
    group_proportional_figure_row = []
    group_fpp_row = []
    group_eigen_factor_row = []
    
    for startEnd in list_startEnd:
            start_index = startEnd[0]
            end_index = startEnd[1]
            
            intevals.append([start_index/float(num_valid_paper), end_index/float(num_valid_paper)])
            group_proportional_figure_row.append(np.mean(raw_proportional_figure[start_index:end_index]))
            group_fpp_row.append(np.mean(raw_figure_per_page[start_index:end_index]))
            group_eigen_factor_row.append(np.mean(raw_eigen_factor[start_index:end_index]))
            
    group_proportional_figure.append(group_proportional_figure_row)
    group_fpp.append(group_fpp_row)
    group_eigen_factor.append(group_eigen_factor_row)

    
print types
print intevals
print group_proportional_figure
print group_fpp
print group_eigen_factor
    
x = np.arange(len(group_proportional_figure))
buffer = 0.1
width = 0.15
fig, ax = plt.subplots()

for i in range(0, 4):
    
    values = group_proportional_figure[i]
#     values = group_fpp[i]
    ax.bar(x  + buffer + i*width, values, width, color=type_colors[i], label = types[i])

# add some text for labels, title and axes ticks
fontSize = 18
ax.set_ylabel('Proportion of Figure Types', fontsize = fontSize)
# ax.set_ylabel('|Figures| / |Page|', fontsize = fontSize)
ax.set_xlabel('Ranked Impact by Eigenfactor', fontsize = fontSize)
ax.set_title('Papers from Pubmed Central', fontsize = fontSize)
ax.set_xticks(x + buffer + 2*width)
ax.set_xticklabels( ('Top 5%', '5th to 26th Percentile', '26th to 43th Percentile', 'Bottom 57%') , fontsize = fontSize)
plt.tick_params(axis='both', which='major', labelsize = fontSize)

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels, fontsize = fontSize)
        

plt.show()
fig.savefig('/Users/sephon/Desktop/viz/all_single_new/4Groups_PF_FilterPG_GroupByEF_PP_fileter_PLoS_One_1997-2014_filter_PLos_One.eps', format='eps')

        