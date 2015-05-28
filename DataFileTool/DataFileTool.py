import os, errno
import datetime
import csv

class DataFileTool():
    
    @staticmethod
    def makeDir(dst, dirName):
        path = os.path.join(dst, dirName)        
        try:
            os.makedirs(path)
            print "Create new directory " + path
            return path    
        except OSError as exc: # Python >2.5
            if exc.errno == errno.EEXIST and os.path.isdir(path):
                print  path + ' is existed.'
                pass
            else: raise
            return path
        
    @staticmethod
    def getModelPath(path, dirName):
        dirName = dirName + datetime.datetime.now().strftime("%Y-%m-%d")
        return DataFileTool.makeDir(path, dirName)
    
    
    @staticmethod
    def getFileNameAndSuffix(filePath):
        filename = filePath.split('/')[-1]
        suffix = filename.split('.')[1]
        return filename, suffix
    
        # File format: paper_id_image_id.{png, jpg,....}
    @ staticmethod
    def getIDsFromFilename(filename):
        filename = filename.split('.')
        filename = filename[0].split('_')
        return filename[1], filename[3]
    
    @ staticmethod
    def saveCSV(path, filename, content = None, header = None, mode = 'wb', consoleOut = True):

        if consoleOut:
            print 'Saving image information...'
        filePath = os.path.join(path, filename) + '.csv'
        with open(filePath, mode) as outcsv:
            writer = csv.writer(outcsv, dialect='excel')
            if header is not None:
                writer.writerow(header)
            if content is not None:
                for c in content:
                    writer.writerow(c)
        
        if consoleOut:  
            print filename, 'were saved in', filePath, '\n'
    