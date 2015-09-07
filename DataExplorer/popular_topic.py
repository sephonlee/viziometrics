import csv
import numpy as np
import random
import math
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick

file_name = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/popular_topic_over_year_all.csv'
outFile = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/popular_topic_over_year_all_modified.csv'
outPerFile = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data/popular_topic_over_year_all_per_modified.csv'

data = []
topics = []
last_row = []
count = 0
year = 0
with open(file_name ,'rb') as incsv:
    reader = csv.reader(incsv, dialect='excel')
    reader.next()
    for row in reader:
#         print row
        if row[0] != year:
            year = row[0]
            count = 0
            # The last row
            if len(data) != 0:
                data.append(last_row)
                last_row = []
          
        if count < 3:
            data.append(row)
        elif count == 3:
            last_row = [row[0], 'others', int(row[2])]
        else:
            last_row [2] += int(row[2])

        count += 1
        
data.append(last_row)
print data


for row in data:
    if row[1] not in topics:
        topics.append(row[1])
    
print len(topics)
print topics

finalData = []
finalPerData = []
year = 0
for row in data:
    if year != row[0]:
        newRow = [int(row[0])]
        for topic in topics:
            if topic == row[1]:
                newRow.append(int(row[2]))
            else:
                newRow.append(0)
        
        year = row[0]
        finalData.append(newRow)       
    else:
        newRow = finalData[-1]
        index = topics.index(row[1]) + 1
        newRow[index] = int(row[2])
        
for row in finalData:
    num = sum(row) - row[0]
    new_row = [row[0]]
    for i in range(1, len(row)):
        new_row.append(float(row[i])/num)
    
    finalPerData.append(new_row)
        
        
#     count += 1
#     
#     if count > 3:
#         count = 0

print finalData
print finalPerData
header = topics
header.insert(0, "year")
print header


with open(outFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, dialect='excel')
    writer.writerow(header)
    for row in finalData:
        writer.writerow(row)
        
with open(outPerFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, dialect='excel')
    writer.writerow(header)
    for row in finalPerData:
        writer.writerow(row)