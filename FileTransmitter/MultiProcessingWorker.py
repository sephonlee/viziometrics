import sys
sys.path.append("..")
from Extractor.Common import *
from Extractor.TarParser import *
import multiprocessing as mp
    
def listener(name, q, outPath, outFilename):
    print '%s Listener set up in %s' % (name, mp.current_process().name)
    startTime = time.time()
    outFilePath = os.path.join(outPath, outFilename) + '.csv'
    count = 0
    outcsv = open(outFilePath, 'ab')
    writer = csv.writer(outcsv, dialect = 'excel')
    while True:
        content = q.get()
        if content is not None:
            count += 1
        # Stop
        if content == 'kill':
            costTime = time.time() - startTime
            print'%d tars have been classified in %d sec. Stop Listener in %s\n' % (count - 1, costTime, name)
            break
        
        for row in content:
#             if name == 'Paper_Info':
#                 print 'row', row
            writer.writerow(row)
        if count % 10 == 0 and count != 0:
            print '%d tars have been collected in %s.' % (count, outFilePath)
            
    outcsv.flush()
    outcsv.close()


def extract(local_tarPath, local_tmp_dir, ftpTarPath, q_paper_result, q_figure_result, q_extractor_error):
    try:
        paper_data, figure_data = TarParser().extractInfo(local_tarPath, local_tmp_dir)
        paper_result = zip([paper_data['pmcid']], [paper_data['pmid']], [paper_data['doi']], [paper_data['longname']], [paper_data['shortname']], [paper_data['title']], [paper_data['num_page']], [paper_data['year_pub']], [paper_data['month_pub']], [paper_data['day_pub']])
        figure_result = []
        for row in figure_data:
            if 'img_id' in row:
                print (row['img_id'], row['pmcid'], row['caption'])
                figure_result.append((row['img_id'], row['pmcid'], row['caption']))
        
        q_paper_result.put(paper_result)
        q_figure_result.put(figure_result)
    except:
        q_extractor_error.put(zip([ftpTarPath]))


    
def upload(bucket, tarName, S3Dir, local_tarPath, ftpTarPath, ftpTarInfo, q_fileOnS3, q_upload_error):
    try:
        upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
#         os.remove(local_tarPath)
        q_fileOnS3.put(zip([os.path.join(S3Dir, tarName)], [ftpTarInfo[1]]))
    except:
        q_upload_error.put(zip([ftpTarPath]))
          
def TarTransmisionWorkerDB(args):
    
    ftpTarInfo, q_ftp_error, q_fileOnS3, q_upload_error, = args
    ftpTarPath = repr(ftpTarInfo[1])[1:-3]
    tarName = ftpTarPath.split('/')[-1]
    local_tarPath = os.path.join(local_tmp_dir, tarName)

    # Downloading
    isDownloaded = False
    try:
        download(ftpTarPath, local_tarPath)
        isDownloaded = True
    except:
        q_ftp_error.put(zip([ftpTarPath], [ftpTarInfo[1]]))
            
    # Uploading to S3
    try:
        if isDownloaded:
            upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
            os.remove(local_tarPath)
            q_fileOnS3.put(zip([os.path.join(S3Dir, tarName)], [ftpTarInfo[1]]))
        else:
            q_upload_error.put(zip([ftpTarPath]))
    except:
        q_upload_error.put(zip([ftpTarPath]))
    finally:
        if os.path.isfile(local_tarPath):
            os.remove(local_tarPath)

def TarTransmisionWorker(args):
    
    ftpTarInfo, q_ftp_error, q_fileOnS3, q_upload_error, = args
#     print 'ftpTarPath', ftpTarInfo
    ftpTarPath =  ftpTarInfo[0]
#     print 'ftpTarPath', ftpTarPath
    tarName = ftpTarPath.split('/')[-1]
    local_tarPath = os.path.join(local_tmp_dir, tarName)

    # Downloading
    isDownloaded = False
    try:
        download(ftpTarPath, local_tarPath)
        isDownloaded = True
    except:
        q_ftp_error.put(zip([ftpTarPath], [ftpTarInfo[1]]))
            
    # Uploading to S3
    try:
        if isDownloaded:
            upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
            os.remove(local_tarPath)
            q_fileOnS3.put(zip([os.path.join(S3Dir, tarName)], [ftpTarInfo[1]]))
        else:
            q_upload_error.put(zip([ftpTarPath]))
    except:
        q_upload_error.put(zip([ftpTarPath]))
    finally:
        if os.path.isfile(local_tarPath):
            os.remove(local_tarPath)
#         os.remove(local_tarPath)

              
def TarExtractingWorker_(args):
    
    ftpTarInfo, q_paper_result, q_figure_result, q_ftp_error, q_extractor_error, q_fileOnS3, q_upload_error, = args
#     print 'ftpTarPath', ftpTarInfo
    ftpTarPath =  ftpTarInfo[0]
#     print 'ftpTarPath', ftpTarPath
    tarName = ftpTarPath.split('/')[-1]
    local_tarPath = os.path.join(local_tmp_dir, tarName)
#     print 'local_tarPath', local_tarPath
    
    tarName = ftpTarPath.split('/')[-1]
    local_tarPath = os.path.join(local_tmp_dir, tarName)
    # Downloading
    try:
        download(ftpTarPath, local_tarPath)
#         extract(local_tarPath, local_tmp_dir, ftpTarPath, q_paper_result, q_figure_result, q_extractor_error)
#         upload(bucket, tarName, S3Dir, local_tarPath, ftpTarPath, ftpTarInfo, q_fileOnS3, q_upload_error)
    except:
#         try:
#             download(ftpTarPath, local_tarPath)
#             extract(local_tarPath, local_tmp_dir, ftpTarPath, q_paper_result, q_figure_result, q_extractor_error)
#             upload(bucket, tarName, S3Dir, local_tarPath, ftpTarPath, ftpTarInfo, q_fileOnS3, q_upload_error)
#         except:
#             try:
#                 download(ftpTarPath, local_tarPath)
#                 extract(local_tarPath, local_tmp_dir, ftpTarPath, q_paper_result, q_figure_result, q_extractor_error)
#                 upload(bucket, tarName, S3Dir, local_tarPath, ftpTarPath, ftpTarInfo, q_fileOnS3, q_upload_error)
#             except:
        q_ftp_error.put(zip([ftpTarPath], [ftpTarInfo[1]]))
    finally:
        os.remove(local_tarPath)
    

        

        
        
                
def TarExtractingWorker(args):
    
    ftpTarInfo, q_paper_result, q_figure_result, q_ftp_error, q_extractor_error, q_fileOnS3, q_upload_error, = args
#     print 'ftpTarPath', ftpTarInfo
    ftpTarPath =  ftpTarInfo[0]
#     print 'ftpTarPath', ftpTarPath
    tarName = ftpTarPath.split('/')[-1]
    local_tarPath = os.path.join(local_tmp_dir, tarName)
#     print 'local_tarPath', local_tarPath
    

    # Downloading
    isDownloaded = False
    try:
        download(ftpTarPath, local_tarPath)
#         print 'Downloading %s from FTP to %s' %(ftpTarPath, local_tarPath) ####
#         ftp = FTP('ftp.ncbi.nlm.nih.gov')     # connect to host, default port
#         ftp.login()
#         ftp.cwd('pub/pmc')   
#         file = open(local_tarPath, 'wb')
#         ftp.retrbinary('RETR %s' % ftpTarPath, file.write)
#         file.close()
        isDownloaded = True
    except:
        q_ftp_error.put(zip([ftpTarPath], [ftpTarInfo[1]]))
    
    # Extracting
#     try:
# #         print 'Extracting info from the tar file'
# #         paper_data, figure_data = TarParser().extractInfo(local_tarPath, local_tmp_dir)
# #         paper_result = zip([paper_data['pmcid']], [paper_data['pmid']], [paper_data['doi']], [paper_data['longname']], [paper_data['shortname']], [paper_data['title']], [paper_data['num_page']], [paper_data['year_pub']], [paper_data['month_pub']], [paper_data['day_pub']])
# #         figure_result = []
# #         for row in figure_data:
# #             if 'img_id' in row:
# #                 figure_result.append((row['img_id'], row['pmcid'], row['caption']))
#                 
#         paper_result, figure_result = extract(local_tarPath, local_tmp_dir)
#         q_paper_result.put(paper_result)
#         q_figure_result.put(figure_result)
#     except:
#         q_extractor_error.put(zip([ftpTarPath]))
        
    # Uploading to S3
    try:
#         upload(bucket, tarName, S3Dir, local_tarPath)
        if isDownloaded:
            upLoadingFileToPath(bucket, tarName, S3Dir, local_tarPath)
            os.remove(local_tarPath)
            q_fileOnS3.put(zip([os.path.join(S3Dir, tarName)], [ftpTarInfo[1]]))
        else:
            q_upload_error.put(zip([ftpTarPath]))
    except:
        q_upload_error.put(zip([ftpTarPath]))
#     finally:
#         os.remove(local_tarPath)
