# Modify .csv files that has "[ ]"

import os, errno, csv
import locale


def read_float_with_comma(num):
    return locale.atof(num)
 
csv_path = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/caption'
figureCSVList = []
paperCSVList = []
errorCSVList = []
 
for dirPath, dirNames, fileNames in os.walk(os.path.join(csv_path, '0506')):   
            for f in fileNames:
                suffix = f.split('.')[1]
                if suffix == 'csv':
#                     print f.split('_')
                    print f
                    type = f.split('_')[1]
                    if type == 'figure':
                        figureCSVList.append(os.path.join(dirPath, f))
                    elif type == 'paper':
                        paperCSVList.append(os.path.join(dirPath, f))
                    elif type == 'extraction':
                        errorCSVList.append(os.path.join(dirPath, f))
                         
print figureCSVList
print paperCSVList
print errorCSVList

finalFigureFile = os.path.join(csv_path, 'finalFigureCaption.csv')
finalPaperFile = os.path.join(csv_path, 'finalPaperInfo.csv')
finalErrorFile = os.path.join(csv_path, 'finalExtractionError.csv')
count_classFile = 0
count_errorFile = 0
count_invalidFile = 0
# a = "6.52353753563e-7"
# print float(a)
 
  
with open(finalFigureFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, delimiter='|')
    header = ['img_id', 'pmcid', 'caption']
    writer.writerow(header)
    for f in figureCSVList:
        print 'converting %s...' %f
        with open(f ,'rb') as incsv:
            reader = csv.reader(incsv, dialect='excel')
            reader.next()
            for row in reader:
                newRow = [row[0], row[1], row[2].replace('"', '')]
                writer.writerow(row)
                 
         
# with open(finalPaperFile, 'wb') as outcsv:
#     writer = csv.writer(outcsv, dialect='excel')
#     header = header = ['pmcid', 'pmid', 'doi', 'longname', 'shortname', 'title', 'num_page', 'year_pub', 'month_pub', 'day_pub']
#     writer.writerow(header)
#     for f in paperCSVList:
#         print 'converting %s...' %f
#         with open(f ,'rb') as incsv:
#             reader = csv.reader(incsv, dialect='excel')
#             reader.next()
#             for row in reader:
#                 writer.writerow(row)
                
# with open(finalErrorFile, 'wb') as outcsv:
#     writer = csv.writer(outcsv, dialect='excel')
#     header = ['file_path', 'key_size'] 
#     writer.writerow(header)
#     for f in errorCSVList:
#         print 'converting %s...' %f
#         with open(f ,'rb') as incsv:
#             reader = csv.reader(incsv, dialect='excel')
#             reader.next()
#             for row in reader:
#                 writer.writerow(row)
#              

#         