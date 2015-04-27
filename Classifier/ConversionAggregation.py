# Modify .csv files that has "[ ]"

import os, errno, csv
import locale


def read_float_with_comma(num):
    return locale.atof(num)
 
csv_path = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/'
resultCSVList = []
errorCSVList = []
 
for dirPath, dirNames, fileNames in os.walk(os.path.join(csv_path, 'conversion')):   
            for f in fileNames:
                suffix = f.split('.')[1]
                if suffix == 'csv':
#                     print f.split('_')
                    type = f.split('_')[1]
                    if type == 'converted':
                        resultCSVList.append(os.path.join(dirPath, f))
                    elif type == 'error':
                        errorCSVList.append(os.path.join(dirPath, f))
                         
print resultCSVList
print errorCSVList
finalClassFile = os.path.join(csv_path, 'conversion.csv')
finalErrorFile = os.path.join(csv_path, 'conversionError.csv')
count_classFile = 0
count_errorFile = 0
# a = "6.52353753563e-7"
# print float(a)
 
with open(finalClassFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, dialect='excel')
    header = ['image_location', 'is_convertable']
    writer.writerow(header)
    for f in resultCSVList:
        print 'converting %s...' %f
        with open(f ,'rb') as incsv:
            reader = csv.reader(incsv, dialect='excel')
            reader.next()
            for row in reader:
                count_classFile += 0
                newRow = [row[0], 1]
                writer.writerow(newRow)
                 
         
with open(finalErrorFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, dialect='excel')
    header = ['image_location', 'is_convertable']
    writer.writerow(header)
    for f in errorCSVList:
        print 'converting %s...' %f
        with open(f ,'rb') as incsv:
            reader = csv.reader(incsv, dialect='excel')
            reader.next()
            for row in reader:
                newRow = [row[0], 0]
                writer.writerow(newRow)
             
             
# csv_path = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/all_s3file/'
# finalClassFile = os.path.join(csv_path, 'finalClass.csv')
# finalErrorFile = os.path.join(csv_path, 'finalError.csv')
# count_classFile = 0
# count_errorFile = 0
# # a = "6.52353753563e-7"
# # print float(a)
# 
# print finalClassFile
# with open(finalClassFile, 'wb') as outcsv:
#     writer = csv.writer(outcsv, dialect='excel')
#     header = ['image_id', 'image_loc', 'file_format', 'file_size']
#     writer.writerow(header)
#     f = "/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/all_s3file/s3_files.csv"
#     print 'converting %s...' %f
#     with open(f ,'rb') as incsv:
#         reader = csv.reader(incsv, dialect='excel')
#         reader.next()
#         for row in reader:
#             count_classFile += 0
#             fname = row[0]
#             image_id = fname.split('/')[1]
#             classname = row[1]
#             classname = classname[2:-2]
#             
#             newRow = [image_id, fname, row[1], row[2]]
#             writer.writerow(newRow)   
#         