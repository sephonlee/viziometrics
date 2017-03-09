import csv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick


###


type_index = 9

#query: year_all_single.sql
types = ['Diagram', 'Photo', 'Data Visualization', 'Table' ]
type_colors = ['#1F77B4', '#7F7F7F', '#C49C94', '#DBDB8D']

file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/forpaper/year_all_single.csv'



year_pub = []
num_papes = []
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
        num_papes.append(int(row[1]))
        eigen_factor.append(float(row[2]))
        avg_num_pages.append(float(row[3]))
        avg_figure_page.append(float(row[4]))
        avg_equations_page.append(float(row[5]))
        avg_tables_page.append(float(row[6]))
        avg_photos_page.append(float(row[7]))
        avg_visualizations_page.append(float(row[8]))
        avg_diagrams_page.append(float(row[9]))
            

fig, ax = plt.subplots(2, sharex=True, sharey=False)
# x = year_pub
# y = eigen_factor

x = np.array(year_pub)
y = (avg_diagrams_page, avg_photos_page, avg_visualizations_page, avg_tables_page)
# y = np.row_stack((avg_diagrams_page, avg_photos_page, avg_visualizations_page, avg_tables_page))

plots = []
fontSize = 18
for i in range(0,4):
    tmp, = ax[0].plot(x[-25:], y[i][-25:], label= types[i], color = type_colors[i], linewidth=2.0)
    plots.append(tmp)
    ax[0].scatter(x[-25:], y[i][-25:], s = 30, marker = 'o', color = type_colors[i])

ax[0].xaxis.set_ticks(range(1990,2015,1))
ax[0].set_xticklabels( ('90\'', '91\'', '92\'', '93\'', '94\'', '95\'', '96\'','97\'', '98\'', '99\'', '00\'', '01\'', '02\'', '03\'', '04\'', '05\'', '06\'', '07\'', '08\'', '09\'', '10\'', '11\'', '12\'', '13\'', '14\'') , fontsize = fontSize - 2)
ax[0].tick_params(axis='both', which='major', labelsize = fontSize - 2)
ax[0].set_ylabel("|Figure| / |Page|", fontsize = fontSize - 2)

ax2 = ax[0].twinx() 
ef_plot, = ax2.plot(x, eigen_factor, '--', label= 'Eigenfactor', color = '#ED1C24', linewidth=2.0)
# ax2.tick_params(axis='y', colors='#ED1C24')
ax2.tick_params(axis='both', which='major', labelsize = fontSize - 2)
ax2.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
ax2.set_ylim([0, 0.0000007])
ax2.set_ylabel("Average Eigenactor", fontsize = fontSize - 2)

plots.append(ef_plot)
types.append('Eigenfactor')
# ax[0].grid()

# plt.legend(plots, types)



npg_plot, = ax[1].plot(x[-25:], avg_num_pages[-25:], 's-', label= 'Avg. Num. of Pages', color = '#D7DF23', linewidth=2.0)
ax[1].xaxis.set_ticks(range(1990,2015,1))
ax[1].set_xticklabels( ('90\'', '91\'', '92\'', '93\'', '94\'', '95\'', '96\'','97\'', '98\'', '99\'', '00\'', '01\'', '02\'', '03\'', '04\'', '05\'', '06\'', '07\'', '08\'', '09\'', '10\'', '11\'', '12\'', '13\'', '14\'') , fontsize = fontSize - 2)
ax[1].tick_params(axis='both', which='major', labelsize = fontSize - 2)
ax[1].set_ylabel("Average Number of Pages", fontsize = fontSize - 2)

plots.append(npg_plot)
types.append('Avg. Number of Pages')

ax3 = ax[1].twinx() 
np_plot, = ax3.plot(x, num_papes, 'd-', label= 'Num. of Papers', color = '#C49A6C', linewidth=2.0)
ax3.tick_params(axis='both', which='major', labelsize = fontSize - 2)
# ax3.tick_params(axis='y', colors='#8B5E3C')
ax3.set_ylabel("Number of Papers", fontsize = fontSize - 2)
# ax3.yaxis.set_major_formatter(mtick.FormatStrFormatter('%.1e'))
ax3.set_ylim([0, 120000])

types.append('Number of Papers')
plots.append(np_plot)

ax[1].set_xlabel("Publishing Year", fontsize = fontSize)

ax[0].set_title('Global Visual Trend', fontsize = fontSize)
plt.legend(plots, types)

# plots.append(np_plot)
# 
# types.append('EigenFactor')
# plt.legend(plots, types)



# f.subplots_adjust(hspace=0)
# plt.setp([a.get_xticklabels() for a in f.axes[:-1]], visible=False)
plt.xlim(1990, 2015)
plt.show()

fig.savefig('/Users/sephon/Desktop/Research/VizioMetrics/Visualization/forpaper2015/Overview_year.eps', format='eps')
            