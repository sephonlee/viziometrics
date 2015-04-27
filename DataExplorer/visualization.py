import csv
import numpy as np
import matplotlib.pyplot as plt

file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/average_ef.csv'
data = []
with open(file_name ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel')
    reader.next()
    for row in reader:
        tmp = {'journal_name': row[0],
               'eigen_factor': float(row[1]),
               'num_papers': int(row[2]),
               'num_figures': int(row[3]),
               'num_equations': int(row[4]),
               'num_tables': int(row[5]),
               'num_photos': int(row[6]),
               'num_visualizations': int(row[7]),
               'num_diagrams': int(row[8]),
               'avg_figure': float(row[9]),}
        data.append(tmp)
        
print len(data)

x = np.zeros([1, len(data)])
print x.shape
y = np.zeros([1, len(data)])
for i, row in enumerate(data):
#     x[0, i] = float(row['num_figures']) / row['num_papers']
    x[0, i] = row['num_equations']/float(row['num_figures'])
    y[0, i] = row['eigen_factor']

print x
print y
plt.scatter(x, y, alpha=0.5)
plt.axis([0, 1, 0, 0.0000007])
plt.show()

        