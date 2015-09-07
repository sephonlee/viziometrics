import csv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


###

journals = ['The_Journal_of_Cell_Biology', 'The_Journal_of_Experimental_Medicine', 'Nucleic_Acids_Research', 'PLoS_One']
# journal = journals[0]
type_index = 9


f, ax = plt.subplots(4, sharex=True, sharey=False)
ef_plot = None
for i in range(0,4):
    
    journal = journals[i]

    types = ['Diagram', 'Photo', 'Data Visualization', 'Table' ]
    type_colors = ['#1F77B4', '#7F7F7F', '#C49C94', '#DBDB8D']
    
    file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/year_all_single_%s.csv' % journal
    
    
    
    year_pub = []
    num_figures = []
    eigen_factor = []
    avg_num_pages = []
    avg_figure_page = []
    avg_equations_page = []
    avg_tables_page = []
    avg_photos_page = []
    avg_visualizations_page = []
    avg_diagrams_page = []
    
    with open(file_name ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        for row in reader:
                
            year_pub.append(int(row[0]))
            num_figures.append(int(row[1]))
            eigen_factor.append(float(row[2]))
            avg_num_pages.append(float(row[3]))
            avg_figure_page.append(float(row[4]))
            avg_equations_page.append(float(row[5]))
            avg_tables_page.append(float(row[6]))
            avg_photos_page.append(float(row[7]))
            avg_visualizations_page.append(float(row[8]))
            avg_diagrams_page.append(float(row[9]))
                
    
#     fig, ax = plt.subplots()
    # x = year_pub
    # y = eigen_factor
    x = np.array(year_pub)
    y = np.row_stack((avg_diagrams_page, avg_photos_page, avg_visualizations_page, avg_tables_page))
    ax[i].stackplot(x, y, colors = type_colors)
    
    #####
    ax2 = ax[i].twinx() 
    ef_plot = ax2.plot(x, eigen_factor, '--', label= 'Average EigenFactor', color = '#594A42', linewidth=2.0)
    ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
    ax2.set_ylim([0, max(eigen_factor)*1.1])
    #####
#     ax2.tick_params(axis='y', colors='#594A42')
    
    fontSize = 18
    
    ax[i].xaxis.set_ticks(range(1997,2015,1))
    ax[i].set_xticklabels( ('97\'', '98\'', '99\'', '00\'', '01\'', '02\'', '03\'', '04\'', '05\'', '06\'', '07\'', '08\'', '09\'', '10\'', '11\'', '12\'', '13\'', '14\'') , fontsize = fontSize - 2)
    ax[i].tick_params(axis='both', which='major', labelsize = fontSize - 2)

#     ax[i].xlim(1997, 2014)


fontSize = 18
rect = [ef_plot[0]]
# rect = []
for color in type_colors:
    rect.append(plt.Rectangle((0, 0), 1, 1, fc = color))

types.insert(0, 'EigenFactor')
plt.legend(rect, types)
plt.legend

# plt.ylabel('|Figure| / Page', fontsize = fontSize)
# plt.xlabel('Year', fontsize = fontSize)
# plt.title('Papers from Pubmed Central', fontsize = fontSize)

f.text(0.5, 0.95, 'Papers from Pubmed Central', ha='center', fontsize = fontSize)
f.text(0.5, 0.04, 'Year', ha='center', fontsize = fontSize)
f.text(0.08, 0.5, '|Figure| / |Page|', va='center', rotation='vertical', fontsize = fontSize)
f.text(0.95, 0.5, 'Average EigenFactor', va='center', rotation='vertical', fontsize = fontSize)

        
plt.xlim(1997, 2014)

# f.subplots_adjust(hspace=0)
plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)

plt.show()
f.savefig('/Users/sephon/Desktop/viz/all_single_new/4Journals_Stackline_year.eps', format='eps')
            