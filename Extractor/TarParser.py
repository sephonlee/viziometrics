import os
import tarfile
import csv
import re
import shutil
import xml.etree.ElementTree as ET
from cStringIO import StringIO
import multiprocessing as mp
from pdfminer.pdfpage import PDFPage
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfdocument import PDFDocument

def getTarFilesFromPah(path):
    fileList = []
    num = 0;
    for dirPath, dirNames, fileNames in os.walk(path):   
            for f in fileNames:
                if f.endswith("tar.gz"):
                    fileList.append(os.path.join(dirPath, f))
                    num += 1
    return fileList


def csvFigureSaver(csv_file_path, figure_data):
    csv_file_path = os.path.join(csv_file_path, 'pubmed_figure_caption_rest.csv')
    outcsv = open(csv_file_path, 'ab')
    row = figure_data
    writer = csv.writer(outcsv, dialect = 'excel')
    for row in figure_data:
        if 'img_id' in row:
            result = (row['img_id'], row['pmcid'], row['caption'])
            writer.writerow(result)
    outcsv.close()
    return


def csvPaperSaver(csv_file_path, paper_data):
    csv_file_path = os.path.join(csv_file_path, 'pubmed_paper_info_rest.csv')
    outcsv = open(csv_file_path, 'ab')
    row = paper_data
    writer = csv.writer(outcsv, dialect = 'excel')
#     for row in paper_data:
#         print row

    if row is not None:
        result = (row['pmcid'], row['pmid'], row['doi'], row['longname'], row['shortname'], row['title'], row['num_page'], row['year_pub'], row['month_pub'], row['day_pub'])
#         result = zip([row['pmcid']], y_pred, y_proba, [imageFormat], [imDim[0]], [imDim[1]], [key.size])
        writer.writerow(result)
    outcsv.close()
    return

class TarParser():
    
    def __init__(self):
        return
    
    @ staticmethod
    def extractFileFromTarFile(file_path, tmp_save_path):
        
        save_path = None
        file_dic = {}
        extracted_file_path = {}
        extracted_file_path['pdf'] = []

#         try:
        if file_path.endswith("tar.gz"):
            save_path = os.path.join(tmp_save_path, file_path.split('/')[-1].split('.')[-3])
            tar = tarfile.open(file_path)
            members = tar.getmembers()
            for m in members:
                filename = m.name.split('.')
                extension = filename[-1].lower()

                if extension == 'pdf':
                    tar.extract(m, path=tmp_save_path)
                    extracted_file_path['pdf'].append(os.path.join(tmp_save_path, m.name))
                elif extension == 'nxml':
                    tar.extract(m, path=tmp_save_path)
                    extracted_file_path[extension] = os.path.join(tmp_save_path, m.name)
                elif extension != 'gif':
                    if len(filename) > 1:
                        key = m.name[:-len(extension)-1].split('/')[-1]
                    else:
                        key = filename[0]
                    file_dic[key] = extension        
            tar.close()
#         except:
#             print file_path, "is not extracted successfully"
#         finally:
        return extracted_file_path, file_dic, save_path;
    
    @ staticmethod
    def getInfo(extracted_file_path, file_dic, save_path):
        
#         try:
        ###### From nxml ########
        nxml_file = extracted_file_path['nxml']
        tree = ET.parse(nxml_file)
        root = tree.getroot()
        paper_data = {}
        figure_data = []
        paper_data['doi'] = '\N'
        paper_data['pmid'] = '\N'
        paper_data['pmcid'] = '\N'
        paper_data['title'] = '\N'
        paper_data['num_page'] = '\N'
        paper_data['publisher-id'] = '\N'
        paper_data['day_pub'] = '\N'
        paper_data['year_pub'] = '\N'
        paper_data['month_pub'] = '\N'
        paper_data['longname'] = '\N'
        paper_data['shortname'] = '\N'
        
        # find publisher
        n = root.findall('.//journal-title')
        if n and n[0].text is not None:
            paper_data['longname'] = TarParser.replaceUnicodeChar(n[0].text.encode('utf-8'))
        
        n = root.findall('.//journal-id')
        if n and n[0].text is not None:
            paper_data['shortname'] = TarParser.replaceUnicodeChar(n[0].text.encode('utf-8'))
        
        # paper title
        n = root.findall('.//article-title')
        if n :
            if n[0].text is not None:
                paper_data['title'] = TarParser.replaceUnicodeChar(n[0].text.encode('utf-8'))
        
        # find pub-id
        for n in root.findall('.//article-id'):  
            if n.attrib:
                att = n.attrib[n.attrib.keys()[0]]
                if att == 'pmc':
                    pmcid = n.text
                    if pmcid[0:3].lower() == 'pmc':
                        paper_data['pmcid'] = pmcid[3:]
                    else:
                        paper_data['pmcid'] = n.text
                else:
                    paper_data[att] = n.text
           
        # pub date
        n = root.findall('.//pub-date//year')
        if n and n[0].text is not None:
            paper_data['year_pub'] = n[0].text
        
        n = root.findall('.//pub-date//day')
        if n and n[0].text is not None:
            paper_data['day_pub'] = n[0].text
        
        n = root.findall('.//pub-date//month')
        if n and n[0].text is not None:
            paper_data['month_pub'] = n[0].text
         
        for n in root.findall('.//fig'):
    
            fig_info = {}
            
            # id
            fig_info['id'] = n.attrib['id']
            # image_id
            g = n.findall('.//graphic')
            if g:
                att = g[0].attrib[g[0].attrib.keys()[0]]
                try:
                    fig_info['img_id'] = 'PMC' + paper_data['pmcid'] + '_' + att + '.' + file_dic[att]
                except:
                    print 'Do not find the images in the tarfile'
            
            # paper_id
            fig_info['pmcid'] = paper_data['pmcid']
            
            # caption
            caption_nodes = n.findall('.//caption')
            if caption_nodes:
                caption = TarParser.getFullCaptionContent(caption_nodes[0])
                caption = caption.replace("'", '"').strip()
                if len(caption) > 0 and caption[0] == '"':
#                     print caption
                    caption = caption[2:-1]
                fig_info['caption'] = caption
            else:
                fig_info['caption'] = '\N'
            
            figure_data.append(fig_info)
        
        ###### From pdf ########
        pdfs = extracted_file_path['pdf']        
        nxml_file_name = nxml_file.split('/')[-1][:-5]            
        
        if pdfs:
            paper_pdf = None
            if len(pdfs) > 1:
                for pdf in pdfs:
                    pdf_name = pdf.split('/')[-1][:-4]
                    if pdf_name == paper_data['publisher-id'] or pdf_name == nxml_file_name:
                        paper_pdf = pdf
            else:
                paper_pdf = pdfs[0]
            
            if paper_pdf is not None:
                try:
                    fp = open(paper_pdf, 'rb')
                    page_count = 0
                    for page in PDFPage.get_pages(fp, []):
                        page_count += 1
                        
                    paper_data['num_page'] = page_count
                except:
                    print 'pdf not allowed to extract'
        
        return paper_data, figure_data
#         except:
#             print 'Extracted Error'
#             return None, None
  
#         return paper_data, figure_data
    
    @ staticmethod
    def replaceUnicodeChar(string):
        try:
            string = string.decode('unicode_escape').encode('ascii','ignore')
    #         string = string.encode('utf-8')
            string = re.sub(r'\s+', ' ', string)
    #         except:
    #             string = repr(string)
    #         string = " ".join(string.split())
    #         string = string.replace('u\'' ,'')
    #         string = re.sub('\W(.\w+)\d.|\W(u\w+)\d', '', string)
    #         string = re.sub('.\W(\w+)\d.|\W(u\w+)\d', '', string)
    #         string = string.replace('\'' ,'')
            return string
        except:
            return '[unknown char]'
    
    
    @ staticmethod
    def getFullCaptionContent(node):
        str_buff = StringIO()
        full_text = TarParser.getFullCaptionContent_help(node, str_buff).getvalue()
#         print full_text
#         full_text = TarParser.replaceUnicodeChar(full_text) 
        return full_text
    
    @ staticmethod
    def getFullCaptionContent_help(node, str_buff):
        if node.text is not None:
            string = TarParser.replaceUnicodeChar(node.text.encode('utf-8'))
            str_buff.write(string)
        for child in node.getchildren():
            TarParser.getFullCaptionContent_help(child, str_buff)
        if node.tail is not None:
            string = TarParser.replaceUnicodeChar(node.tail.encode('utf-8'))
            #repr not used anymore
            str_buff.write(string)
        return str_buff
    
    @ staticmethod  
    def delTmpFiles(extracted_file_path, save_path): 
        for key in extracted_file_path.keys():
            if key == 'nxml':
                os.remove(extracted_file_path[key])
            else:
                for pdf in extracted_file_path[key]:
                    os.remove(pdf)
        os.rmdir(save_path)
        return
    
    @ staticmethod
    def extractInfo(file_path, tmp_save_path):
        paper_data = None
        figure_data = None
        extracted_file_path, file_dic, save_path = TarParser.extractFileFromTarFile(file_path, tmp_save_path)
        if 'nxml' in extracted_file_path:
            paper_data, figure_data = TarParser.getInfo(extracted_file_path, file_dic, save_path)
#         if 'nxml' in extracted_file_path or len(extracted_file_path['pdf']) > 0:
#             TarParser.delTmpFiles(extracted_file_path, save_path)
#         print 'save paht', save_path, os.path.isfile(save_path)
        if os.path.isdir(save_path):
            shutil.rmtree(save_path)
        return paper_data, figure_data
    
    
# def extractInfoFromTarFile(file_path, tmp_save_path, csv_file_path):
#     nxml_path, save_path = extractNxmlFromTarFile(file_path, tmp_save_path);
#     paper_data = getInfo(nxml_path);
#     
#     outcsv = open(csv_file_path, 'ab')
#     writer = csv.writer(outcsv, dialect = 'excel')
#     writer.writerow(data)
    
if __name__ == '__main__':  
#     tar_file_path = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Source_paper/'
    tar_file_path = '/Users/sephon/Documents/workspace/VizClassification/FileTransmitter/tmp'
    tmp_save_path = '/Users/sephon/Desktop/Research/VizioMetrics/Corpus/Source_paper/temp'
    csv_path = '/Users/sephon/Desktop/Research/VizioMetrics/cloud_result/caption'
    tar_file_list = getTarFilesFromPah(tar_file_path)
#     tar_file_list.remove('/Users/sephon/Desktop/Research/ReVision/source_papers/Pubmed/ee/4d/PLoS_Genet_2012_Apr_12_8(4)_e1002626.tar.gz')
    
#     outcsv = open(os.path.join(csv_path, 'pubmed_paper_info_rest.csv'), 'wb')
#     writer = csv.writer(outcsv, dialect = 'excel')
#     header = ['pmcid', 'pmid', 'doi', 'longname', 'shortname', 'title', 'num_page', 'year_pub', 'month_pub', 'day_pub']
#     writer.writerow(header)
#     outcsv.close()
#     
#     outcsv = open(os.path.join(csv_path, 'pubmed_figure_caption_rest.csv'), 'wb')
#     writer = csv.writer(outcsv, dialect = 'excel')
#     header = ['img_id', 'pmcid', 'caption']
#     writer.writerow(header)
#     outcsv.close()
         
    for i, file_path in enumerate(tar_file_list):
        print i, file_path
        paper_data, figure_data = TarParser.extractInfo(file_path, tmp_save_path)
        if paper_data is not None:
            csvPaperSaver(csv_path, paper_data)
        if figure_data is not None:
            csvFigureSaver(csv_path, figure_data)
    
#     file_path = '/Users/sephon/Documents/workspace/VizClassification/FileTransmitter/tmp/Clin_Oral_Investig_2012_Feb_2_16(1)_109-115.tar.gz'
#     paper_data, figure_data = TarParser.extractInfo(file_path, tmp_save_path)
#     csvFigureSaver(csv_path, figure_data)
#     print paper_data
#     print figure_data
    
#     csv_path = '/Users/sephon/Desktop/Research/VizioMetrics/class_result/paper_info.csv'
#     csvSaver(csv_path, paper_data)
#     paper_data, figure_data = TarParser.extractInfo(file_path, tmp_save_path)
#     print paper_data
#     print figure_data
#     extracted_file_path, file_dic, save_path = TarParser.extractFileFromTarFile(file_path, tmp_save_path)
#     print extracted_file_path
#     print save_path
#     print file_dic
#     TarParser.getInfo(extracted_file_path, file_dic, save_path)
#     TarParser.delTmpFiles(extracted_file_path, save_path)
# print save_path

