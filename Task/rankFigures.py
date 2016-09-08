# -*- coding: utf-8 -*-
import sys
from _AE import Error
sys.path.append("..")
from DataFileTool.DataFileTool import *
from LatestModels import *
from NLP.TFIDF import NLPModel

## use TFIDF score
def getArticleText(items):
    
    text = []
    ids = []
    if len(items) > 0:

#         print items[0][3].encode('utf-8').strip()
#         print items[0][4].strip()
        str = unicode(items[0][4], errors='ignore')
#         print str
#         print str.encode('utf-8').strip();
        
        
        title = items[0][3].strip()
        
        abstract = items[0][4].strip()
        
#         print abstract
#         print unicode(abstract)
         
        abstract = abstract.replace('\t', ' ')
        abstract = abstract.replace('\n', ' ')
        
        title = title.replace('\t', ' ')
        title = title.replace('\n', ' ')
        
#         print abstract
         
        title = unicode(title, errors='ignore')
        abstract = unicode(abstract, errors='ignore')

         
        title = title.encode('utf-8').strip();
        abstract = abstract.encode('utf-8').strip();

        text.append(title + '. ' + abstract)
        ids.append([int(items[0][0]), int(items[0][1]), 'abstract'])
        
    for item in items:
        
        if item[5] is not None:
            text.append(item[5])
        else:
            text.append("null")
            
        ids.append([int(item[0]), int(item[1]), item[2]])
    
        
#     for i in range(len(text)):
#         print text[i]
#         print ids[i]
        
    return text, ids

def rankFigures(query, outPath, offset, limit):
    
    startTime = time.time()
    DBInfoPath = '/Users/sephon/Desktop/Research/VizioMetrics/Database_Information/viziometrics_db_info.txt'
    db_info = ImageDataManager.getDBInfoFromFile(DBInfoPath)
    IDM = ImageDataManager(connectToDB = True, db_info = db_info)

    header = ['pmcid', 'pmid', 'img_id', 'tfidf_score', 'is_repret']    
    csvFilename = 'tfidf_%d-%d' %(offset, offset+limit)
    DataFileTool.saveCSV(outPath, csvFilename, header = header, mode = 'wb', consoleOut = False)


    key_list = IDM.getKeynamesByQuery(query)

    stemmed = True
    
    outFilePath = os.path.join(outPath, csvFilename) + '.csv'
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
        
    for i, k in enumerate(key_list):
        
        if i%10 == 0:
            print "Progress: %d/%d (time: %f)"%(i,limit, time.time()-startTime)

        items = IDM.getKeynamesByQuery('SELECT pmcid, pmid, img_id, title, abstract, caption FROM image_full_table WHERE pmcid = %d;' %int(k[0]))
        documents, ids =  getArticleText(items)
        
        Model = NLPModel()
        Model.create_models(documents, stemmed=stemmed)

         
        similarity = Model.get_similarity_score(documents, stemmed = stemmed)
        similarity_sort = Model.sort_similarity_score(similarity)

        results = similarity_sort[0]
        
        for i, tfidf_result in enumerate(similarity_sort[0]):
            ids[tfidf_result[0]].append(tfidf_result[1])
            if i == 1:
                ids[tfidf_result[0]].append(1)
            else:
                ids[tfidf_result[0]].append(0)
        
        
#         for id in ids:
#             print id
        # save csv
        for i,row in enumerate(ids):
            if i > 0:
                writer.writerow(row)
                
        outcsv.flush()
    
    outcsv.close()
        

if __name__ == '__main__':
        
#     query = 'SELECT s3.img_loc FROM s3_readable_keys as s3, image_composite as ic WHERE s3.img_id = ic.img_id AND s3.is_readable is null AND s3.img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" AND (s3.img_format = "png" or s3.img_format = "jpg") AND ic.is_composite = 1 limit 200000 offset 1300000'
#     query = 'SELECT s3.img_loc FROM s3_readable_keys as s3, image_composite_temp as ic WHERE s3.img_id = ic.img_id AND s3.is_readable is null AND key_size > 0 AND s3.img_id NOT REGEXP "PMC[0-9]+_[a-zA-Z]+[0-9]+-[0-9]+&copy" AND (s3.img_format = "jpg" OR s3.img_format = "png") AND ic.is_composite = 1 limit 100000 offset 30000'
    
    offset = 420000
    limit = 100000
    outPath = "/Users/sephon/Desktop/Research/VizioMetrics/NLP/PubMed_central"
    query = 'SELECT distinct(pmcid) FROM image_full_table ORDER BY pmcid LIMIT %d OFFSET %d;'%(limit, offset)
    rankFigures(query, outPath, offset, limit)
    