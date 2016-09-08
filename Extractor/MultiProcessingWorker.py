from TarParser import *
from Common import *
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
        
        if content is not None:
            if len(content) > 0:
                for row in content:
                    writer.writerow(row)
                if count % 10 == 0 and count != 0:
                    outcsv.flush()
                    print '%d tars have been collected in %s.' % (count, outFilePath)
            
    outcsv.flush()
    outcsv.close()

def TarExtractingWorkerDB(args):
    
    key, q_paper_result, q_figure_result_list, q_extract_error, = args
    
    key = bucket.get_key(key[0])
    num_q_figure_result = len(q_figure_result_list)
    q_figure_result = q_figure_result_list[randint(0, num_q_figure_result-1)]
    
    tarName = key.name.split('/')[-1]
    local_tarPath = os.path.join(local_tmp_dir, tarName)
#     removed = False
    print local_tarPath, mp.current_process().name
    try:
        key.get_contents_to_filename(local_tarPath)
        paper_data, figure_data = TarParser().extractInfo(local_tarPath, local_tar_tmp_dir)
        paper_result = zip([paper_data['pmcid']], [paper_data['pmid']], [paper_data['doi']], [paper_data['longname']], [paper_data['shortname']], [paper_data['title']], [paper_data['num_page']], [paper_data['year_pub']], [paper_data['month_pub']], [paper_data['day_pub']])
        figure_result = []
        for row in figure_data:
            if 'img_id' in row:
                figure_result.append((row['img_id'], row['pmcid'], row['caption']))
                
        q_paper_result.put(paper_result)
        q_figure_result.put(figure_result)
    except:
        q_extract_error.put(zip([key.name], [key.size]))
    finally:
        if os.path.isfile(local_tarPath):
            os.remove(local_tarPath)
        save_path = os.path.join(local_tar_tmp_dir, local_tarPath.split('/')[-1].split('.')[-3])
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
                        
def TarExtractingWorker(args):
    
    key, q_paper_result, q_figure_result_list, q_extract_error, = args
    
    num_q_figure_result = len(q_figure_result_list)
    q_figure_result = q_figure_result_list[randint(0, num_q_figure_result-1)]
    
    tarName = key.name.split('/')[-1]
    local_tarPath = os.path.join(local_tmp_dir, tarName)
#     removed = False
    print local_tarPath, mp.current_process().name
    try:
        key.get_contents_to_filename(local_tarPath)
        paper_data, figure_data = TarParser().extractInfo(local_tarPath, local_tar_tmp_dir)
        paper_result = zip([paper_data['pmcid']], [paper_data['pmid']], [paper_data['doi']], [paper_data['longname']], [paper_data['shortname']], [paper_data['title']], [paper_data['num_page']], [paper_data['year_pub']], [paper_data['month_pub']], [paper_data['day_pub']])
        figure_result = []
        for row in figure_data:
            if 'img_id' in row:
                figure_result.append((row['img_id'], row['pmcid'], row['caption']))
                
        q_paper_result.put(paper_result)
        q_figure_result.put(figure_result)
    except:
        q_extract_error.put(zip([key.name], [key.size]))
    finally:
        if os.path.isfile(local_tarPath):
            os.remove(local_tarPath)
        save_path = os.path.join(local_tar_tmp_dir, local_tarPath.split('/')[-1].split('.')[-3])
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        
