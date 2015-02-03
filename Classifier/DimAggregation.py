# Modify .csv files that has "[ ]"

import os, errno, csv
import locale


def read_float_with_comma(num):
    return locale.atof(num)

csv_path = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/'
resultCSVList = []
errorCSVList = []

for dirPath, dirNames, fileNames in os.walk(os.path.join(csv_path, 'dim_result')):   
            for f in fileNames:
                suffix = f.split('.')[1]
                if suffix == 'csv':
#                     print f.split('_')
                    type = f.split('_')[1]
                    if type == 'class':
                        resultCSVList.append(os.path.join(dirPath, f))
                    elif type == 'error':
                        errorCSVList.append(os.path.join(dirPath, f))
                        
print resultCSVList
print errorCSVList
finalClassFile = os.path.join(csv_path, 'finalDim.csv')
finalErrorFile = os.path.join(csv_path, 'finalDimError.csv')
count_classFile = 0
count_errorFile = 0
# a = "6.52353753563e-7"
# print float(a)

with open(finalClassFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, dialect='excel')
    header = ['image_id', 'image_height', 'image_width']
    writer.writerow(header)
    for f in resultCSVList:
        print 'converting %s...' %f
        with open(f ,'rb') as incsv:
            reader = csv.reader(incsv, dialect='excel')
            reader.next()
            for row in reader:
                count_classFile += 0
                if len(row) == 3:
                    newRow = [row[0], row[1], row[2]]
                    writer.writerow(newRow)
                
        
with open(finalErrorFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, dialect='excel')
    header = ['image_id', 'image_location', 'file_size']
    writer.writerow(header)
    for f in errorCSVList:
        print 'converting %s...' %f
        with open(f ,'rb') as incsv:
            reader = csv.reader(incsv, dialect='excel')
            reader.next()
            for row in reader:
                fname = row[0]
                image_id = fname.split('/')[1]
                row.insert(0, image_id)
                writer.writerow(row)
                
        