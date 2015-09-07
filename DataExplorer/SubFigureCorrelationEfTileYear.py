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

types = ['Table', 'Photo', 'Data Visualization', 'Diagram']
type_colors = ['#DBDB8D', '#7F7F7F', '#C49C94', '#1F77B4']
fig = plt.figure()
ax = fig.add_subplot(1,1,1)

for type_index in range(6, 10):
    finalResult = []              #01                      #07
    group_sizes = [23, 24, 19, 21, 24, 31, 39, 68, 62, 59, 145, 255, 300, 330, 330, 218, 67, 2]
    outFile = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/year_correlation_visualization.csv'
    
#     fig = plt.figure()
#     ax = fig.add_subplot(1,1,1)
    
    for year_pub in range(2013, 2015):
        file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/figure_paper_sub_class_%d_filter_PLoS_One.csv' % year_pub
    
        ###
    #     file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/figure_paper_sub_class_2005_filter_PLoS_One.csv'
        num_bin = 20
        group_size = 1999
        
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
        
    #     group_size = num_valid_paper/200
        group_size = group_sizes[year_pub-1997]
        print 'number of paper: ', num_valid_paper
        print 'group_size:', group_size
        
        raw_proportional_figure = np.zeros([len(data)])
        raw_eigen_factor = np.zeros([len(data)])
        raw_figure_per_page = np.zeros([len(data)])
        raw_page = np.zeros([len(data)])
        raw_figure = np.zeros([len(data)])
        
        tmpEF = 1
        paper_count = 0
        all_EF = {}
        list_paper_count = [] 
        list_change_ef_index = [0]
        list_pdf_paper_count = []
         
        for i, row in enumerate(data):
        
            if (tmpEF - row['eigen_factor'] > 0.000000000001):
        #         print tmpEF
        #         print row['eigen_factor']
                all_EF[tmpEF] = paper_count
                tmpEF = row['eigen_factor']
        #         print "next group"
                if paper_count > group_size:
                    list_change_ef_index.append(i)
                    list_paper_count.append(paper_count)
                    list_pdf_paper_count.append(float(i)/num_valid_paper)
                    paper_count = 0
            
            paper_count += 1
            
            if i == len(data) - 1:
                list_paper_count.append(paper_count)
                list_pdf_paper_count.append(float(i)/num_valid_paper)
                
            
            raw_proportional_figure[i] = row['proportional_figure']
            raw_eigen_factor[i] = row['eigen_factor']
            raw_figure_per_page[i] = row['figure_per_page']
            raw_page[i] = row['num_pages']
            raw_figure[i] = row['num_figures']
        
        num_gorup = len(list_paper_count)
        print "number of group: ", num_gorup
        print list_paper_count
        list_change_ef_index.append(len(data))
        print list_change_ef_index
        print list_pdf_paper_count
        
        num_bin = len(list_change_ef_index) - 1
        
        # print all_EF
        
        row_ranking = getRanking(raw_eigen_factor)
        
        ######## Re-Sorting
        # sort_indices = np.argsort(raw_page, axis=0)  ######### Sort by page (original data sorted by EigenFactor)
        # print sort_indices
        #  # ori_y = y.copy()
        # raw_proportional_figure = raw_proportional_figure[sort_indices][::-1]
        # raw_eigen_factor = raw_eigen_factor[sort_indices][::-1]
        # raw_figure_per_page = raw_figure_per_page[sort_indices][::-1]
        # raw_page = raw_page[sort_indices][::-1]
        # raw_figure = raw_figure[sort_indices][::-1]
        #########
            
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
        for i in range(0, len(list_change_ef_index)-1):
        #     start_index = int(math.ceil((i-1) * capacity))
        #     end_index = int(math.ceil(i * capacity))
        
            start_index = list_change_ef_index[i]
            end_index = list_change_ef_index[i+1]
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
#         y_value = group_proportional_figure
        y_std = group_proportional_figure_std ###########
        
        ## Plot Scatter
        slope, intercept, r_value, p_value, std_err = stats.linregress(x_value, y_value)
        # plt.plot(x_value, x_value*slope+intercept)
        
    #     plt.scatter(x_value, y_value, s = np.sqrt(np.array(list_paper_count)), alpha=0.5)
    #     plt.axis([0, np.max(x_value) * 1.1, 0, np.max(y_value) * 1.1])
    #     # plt.ylabel('Proportion of Figures that are Classified as Diagrams')
    #     plt.ylabel('Average |Diagram| / |Page|')
    #     plt.xlabel('Average EigenFactor')
    #     plt.text(max(x_value)*9/10, max(y_value) * 2 / 12, 'slope: %f' % slope)
    #     plt.text(max(x_value)*9/10, max(y_value) / 12, 'p-value: %f' % p_value)
    #     plt.show()
         
         
        ## Calculate correlation and p_value
        (cor_coef, p_value_cor) = stats.spearmanr(x_value, y_value)
        
        ## Plot Bar Chart
        ## Assign X, Y value
        x_value = group_eigen_factor ##################
        y_value = group_fpp ################
#         y_value = group_proportional_figure
        # y_value -= np.mean(y_value) 
        y_std = group_proportional_figure_std ###########
        
        for y_i in y_value:
            if y_i < 0:
                color.append('#9B7D67')
            else:
                color.append('#B29677')
                
        # index = range(1, num_bin + 1)
        index = np.linspace(0,100, (num_bin))
        print "index1", index
        index = np.array(list_pdf_paper_count) * 100
        print "index2", index
        # bar_width = 1
        bar_width = np.array(list_paper_count)/float(num_valid_paper) * 100
        index -= bar_width
        print "barwidth", bar_width
#         fig = plt.figure()
#         ax = fig.add_subplot(1,1,1)
#          
#          
#          
#          
#         rects1 = ax.bar(index, y_value, bar_width,
#                          color=color,
#         #                 yerr = y_std,
#                          label='Men')
#           
#         print 'x', index
#         print 'y', y_value
#          
#         fontSize = 18
#         fmt = '%.0f%%' # Format the ticks, e.g. '40%'
#         xticks = mtick.FormatStrFormatter(fmt)
#         # plt.ylabel('Number of Paper with Diagram/Figure Greater Then Average')
#         # plt.ylabel('Proportion of Figures that are Classified as Photos')
#         plt.ylabel('Average |Diagram| / |Page|', fontsize = fontSize)
#         plt.xlabel('Ranking via Average EigenFactor', fontsize = fontSize)
# #         fig_title = 'Papers From Pubmed Central in %d' %year_pub
#         plt.title('Papers From Pubmed Central in %d' %year_pub, fontsize = fontSize)
#         # plt.title('Protein Database')
#         plt.text(100*6.5/10, max(y_value) * 19/20, 'correlation coefficient: %f' % cor_coef, fontsize = fontSize)
#         plt.text(100*6.5/10, max(y_value) * 18/20, 'p-value: %f' % p_value_cor, fontsize = fontSize)
#         plt.tick_params(axis='both', which='major', labelsize = fontSize)
#         # for i, count in enumerate(list_paper_count):
#         #     plt.text(i*2, max(y_value) * i/50, count)
#         # plt.axis([-3, 103, np.min(y_value) * 1.1, np.max(y_value) * 1.1])
#         ax.xaxis.set_major_formatter(xticks)
#         print 
#         outFileName = '/Users/sephon/Desktop/viz/all_single_new/%s_FilterPG_GroupByEF_PP_%d_g%d_per%d.eps' %(types[type_index - 6], year_pub, group_size+1, int(index[1]*100))
#         print outFileName
#         
#         plt.show()
#         fig.savefig(outFileName, format='eps')
#     
        result = [year_pub, num_valid_paper, group_size, num_gorup, index[1], cor_coef, p_value_cor]
        finalResult.append(result)
    
    x = []
    y = []
    s = []
    marker = []
    for result in finalResult:
        x.append(result[0])
        y.append(result[5])
        
        if result[6] < 0.05:
            s.append((1-result[6])*30)
            marker.append('o')
        else:
            s.append(1*25)
            marker.append("x")
        
        print result
    #         
    
    
    # with open(outFile, 'wb') as outcsv:
    #     writer = csv.writer(outcsv, dialect='excel')
    #     header = header = ['year_pub', 'num_paper', 'group_size', 'num_group', 'percentile', 'correlation_coefficient', 'p_value']
    #     writer.writerow(header)
    #     for result in finalResult:
    #         writer.writerow(result)
    
    # line, = plt.plot(x, y, label = "test")
    # plt.legend(handles=[line])
    # plt.show()
    

    ax.plot(x, y, label= types[type_index - 6], color = type_colors[type_index - 6], linewidth=2.0)
    for i in range(0, len(y)):
        ax.scatter(x[i], y[i], s = s[i], marker = marker[i], color = type_colors[type_index - 6])

handles, labels = ax.get_legend_handles_labels()
ax.legend(handles, labels)
ax.xaxis.set_ticks(range(1997,2015,1))
fontSize = 18
plt.ylabel('Correlation Coefficient', fontsize = fontSize)
plt.xlabel('Year', fontsize = fontSize)
plt.tick_params(axis='both', which='major', labelsize = fontSize - 2)
plt.xlim(1996, 2015)
plt.title('Correlation between Figures per Page and Paper Impact in Time Series', fontsize = fontSize)
# plt.title('Correlation between Proportion of Figure Types and Paper Impact in Time Series', fontsize = fontSize)
plt.show()
fig.savefig('/Users/sephon/Desktop/viz/all_single_new/All_Type_Year_FPP_FilterPG_GroupByEF_PP_1997-2014_fileter_PLoS_One.eps', format='eps')
                