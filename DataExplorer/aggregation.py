# Modify .csv files that has "[ ]"

import os, errno, csv
import locale


def read_float_with_comma(num):
    return locale.atof(num)
 
csv_path = '/Users/sephon/Desktop/Research/VizioMetrics/Visualization/data'
inputFile = os.path.join(csv_path, 'paper_ranking.csv')
finalFigureFile = os.path.join(csv_path, 'paper_ranking_group.csv')

count_classFile = 0
# a = "6.52353753563e-7"
# print float(a)
 
index = 0

with open(finalFigureFile, 'wb') as outcsv:
    writer = csv.writer(outcsv, delimiter=',')
    header = ['group_id', 'eigen_factor', 'num_figures', 'num_pages','num_equations', 'num_tables','num_photos', 'num_visualizations', 'num_schemes']
    writer.writerow(header)
    with open(inputFile ,'rb') as incsv:
        reader = csv.reader(incsv, dialect='excel')
        reader.next()
        count = 0
        ef = 0
        n_fig = 0
        n_page = 0
        n_eq = 0
        n_tb = 0
        n_pho = 0
        n_viz = 0
        n_sche = 0
        
        for row in reader:
            index += 1
            count += 1
            
            if float(row[3]) > 1:
                print index, 'ef', row[2], row[3]
            
            ef += float(row[3])
            n_fig += int(row[4])
            n_page += int(row[5])
            n_eq += int(row[6])
            n_tb += int(row[7])
            n_pho += int(row[8])
            n_viz += int(row[9])
            n_sche += int(row[10])
            
            if count % 10 == 0:
                newRow = [index, ef, n_fig, n_page, n_eq, n_tb, n_pho, n_viz, n_sche]
#                 print count, index, newRow
                writer.writerow(newRow)
                count = 0
                ef = 0
                n_fig = 0
                n_page = 0
                n_eq = 0
                n_tb = 0
                n_pho = 0
                n_viz = 0
                n_sche = 0
         
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