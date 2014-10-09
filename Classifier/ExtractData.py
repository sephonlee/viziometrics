import os, errno
from os.path import *
import tarfile
import types
import shutil
import csv

VALID_IMAGE_TYPE =  ('jpg', 'tif', 'bmp', 'png', 'tiff')

class ImageFromFile:
    
        
    def __int__(self, path):
        self.path = path    
        
    def __str__(self):
        return self
    
    def setPath(self, path):
        self.path = path
        
    def getPath(self):
        return self.path

    @staticmethod
    # return all file paths from the given directory with given file Type
    # path: directory path
    # type: file type, ex: pdf, tar.gz
    def getFileNamesFromPath(path, fileType):
        print "Get file names from ", path
        fileList = []
        fileType = "." + fileType
        num = 0;
        for dirPath, dirNames, fileNames in os.walk(path):    
            for f in fileNames:
                if f.endswith(".tar.gz"):
                    fileList.append(os.path.join(dirPath, f))
                    num += 1
                    
        
#         self.fileList = fileList
        print num, "files were found"
        return fileList
        
    @staticmethod
    # extract image file from tar file
    def extractTarFromFileName(fileName, extractTarPath, sizeThreshold = 0, validImageType = VALID_IMAGE_TYPE):
        try:
    #         print sizeThreshold
            if isinstance(fileName, types.StringTypes):
                # Extract files from *.tar.gz
                tfile = tarfile.open(fileName)
                if tarfile.is_tarfile(fileName):
                    numExt = 0;
                    numAll = 0;
                    print "Extracting " + fileName + " ...." 
                    for tarinfo in tfile:
    #                     print tarinfo.name
    #                     print tarinfo.size
                        if tarinfo.size > sizeThreshold:
                            name = tarinfo.name.split('.')
                            surfix = name[len(name) - 1]
                            if surfix in validImageType :
                                tfile.extract(tarinfo.name, extractTarPath)
                                numExt += 1
                        numAll += 1
                    print numExt , "/" , numAll , " files were extracted from " , fileName 
                else:
                    print fileName + " is not a tarfile."
                tfile.close()
            else:
                print "Input fileName is invalid"
        except:
            print "Fail to extract " + fileName

    @staticmethod
    # extract image file from tar file list
    def extractTarFromFileList(fileList, extractTarPath, sizeThreshold = 0, validImageType = ('jpg', 'gif', 'tif', 'bmp', 'png', 'tiff')):
        if isinstance(fileList, types.ListType):
            # Extract files from *.tar.gz           
            for f in fileList:
                ImageFromFile.extractTarFromFileName(f, extractTarPath, sizeThreshold)
            print "All files were extracted in " + extractTarPath 
        else:
            print "Input fileList is invalid"

    @staticmethod
    def gatherAllFile(src, dst, catList, catNum, csvFileName, isMultiOut):
        print "Moving files from " + src
        i = 1;
        
        # make output directories
        if isMultiOut:
            for i in range(0, catNum):
                ImageFromFile.makeDir(dst, catList[i])
            
        csvFile = open(csvFileName,'wb')
        csvWriter = csv.writer(csvFile, dialect='excel')
    
        dirIndex = 1
        for dirPath, dirNames, fileNames in os.walk(src):
            
            fileIndex = 1
            for f in fileNames:
                fileName = f.split('.')
                fileName = 'paper_%d_image_%d.' % (dirIndex, fileIndex) + fileName[-1]     
                if isMultiOut:
                    outDir = ImageFromFile.getOutPutDir(i % catNum, catList)
                    newDst = dst + outDir + "/" + fileName
                    print newDst
                else:
                    newDst = dst + fileName
                    print newDst
                shutil.move(os.path.join(dirPath, f), newDst)
    #           print dirNames.
                csvRow = [fileName, dirPath]
                csvWriter.writerow(csvRow)
                i += 1
                fileIndex += 1
            dirIndex += 1
                
        csvFile.close()
        print "All files were moved to " + dst
        # delete rest files and folders
        for dirPath, dirNames, fileNames in os.walk(src):
            for name in fileNames:
                os.remove(os.path.join(dirPath, name))
            for name in dirNames:
                print os.path.join(dirPath, name)
                os.rmdir(os.path.join(dirPath, name))
        # clear src folder after finishing
        os.rmdir(src)
        print "Removed " + src
        
        
    @staticmethod
    def makeDir(dst, dirName):
        path = dst + dirName
        try:
            os.makedirs(path)
            print "Directory " + dirName + " has been made in " + dst 
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                pass
            else: raise
        
    @staticmethod
    def getOutPutDir(index, categories):
        if index < len(categories):
            return categories[index]
        

if __name__ == '__main__':   
    
    path = "/Users/sephon/Desktop/Research/ReVision/source_papers/Pubmed/ee_sub/"
    f = '/Users/sephon/Desktop/Research/ReVision/source_papers/Pubmed/ee/4d/PLoS_Genet_2012_Apr_12_8(4)_e1002626.tar.gz'
    extractTarPath = '/Users/sephon/Desktop/Research/ReVision/source_papers/Pubmed/extImages/'
    
    catList = ['cat_0', 'cat_1', 'cat_2', 'cat_3', 'cat_4', 'cat_5', 'cat_6'];
    
    fileList = ImageFromFile.getFileNamesFromPath(path, "tar.gz")
    
#     csvFileName = extractTarPath + 'fileIndex.csv'
#     with open(csvFileName, 'wb') as csvFile:
#         writer = csv.writer(csvFile)
#         writer.writerows(catList)
    
    
    
#     ImageFromFile.extractTarFromFileName(f, extractTarPath, 8000)


    ImageFromFile.extractTarFromFileList(fileList, extractTarPath, 8000)

    
    dst = "/Users/sephon/Desktop/Research/VizioMetrics/Corpus/"
    csvFileName = dst + "index.csv"
    ImageFromFile.gatherAllFile(extractTarPath, dst, catList, 4, csvFileName, False)


        