import MySQLdb
import json
import pickle
import os, errno
from os.path import *

class ImageDataManager:
    
    db = None
    db_connected = False
    localDataPath = None
    classDataID = {}
    classDataName = {}
    
    def __init__(self, connectToDB = False, db_info = None):
        
        if connectToDB & (db_info is not None):
            host = db_info['host']
            db_username = db_info['db_username']
            db_password = db_info['db_password']
            db_name = db_info['db_name']
            
            self.loginDB(host, db_username, db_password, db_name)
            self.db_connected = True
#         elif localDataPath is not None:
#             self.localDataPath = localDataPath
        else:
            print 'Warning! Don\'t find any data support!'
            
    # login mysql
    def loginDB(self, host, db_username, db_password, db_name):
        self.db = MySQLdb.connect(host, db_username, db_password, db_name)
    
    @ staticmethod
    def getDBInfoFromFile(path):
        f = open(path, 'r')
        db_info = { 'host': f.readline()[0:-1],
                'db_username': f.readline()[0:-1],
                'db_password': f.readline()[0:-1],
                'db_name': f.readline()[0:-1]}
        return db_info
    
    def getKeynamesByQuery(self, query):
        cursor = self.db.cursor()
        cursor.execute(query)
        return cursor.fetchall()
    
    def getKeynames(self, formats = None, size_limit = None, offset = None):
        cursor = self.db.cursor()
        sql = "SELECT img_loc FROM keys"
        
        num_AND = 0
        
        if (formats is not None):
            num_AND += 1
        if (size_limit is not None):
            num_AND += 1
            
        if num_AND >= 0:
            sql = sql + " WHERE"
        
        if size_limit is not None:
            sql = sql + " file_size <= %d" % size_limit
            num_AND -= 1
            if num_AND > 0:
                sql += " AND"
    
        if formats is not None:
            if len(formats) > 0:
                substring = " ("
            
            for i, format in enumerate(formats):
                if i == 0:
                    substring += " format = '%s'" %format
                else:
                    substring += " OR format = '%s'" %format
                    
            if len(formats) > 0:
                substring += ")"
            
            sql += substring
            num_AND -= 1
            
            if num_AND > 0:
                sql += " AND" 
            
                
        if offset is not None:
            sql += " OFFSET %d" % offset
    
        cursor.execute(sql)
        return cursor.fetchall()
    
    @ staticmethod
    def extendPath(path, newFileName):
        if path[-1] in ('/', '\\'):
            return path + newFileName
        else:
            return path + '/' + newFileName
    
    def class_id2class_name(self, class_id):    
        if len(self.classDataID) == 0:
            self.loadClassData()
            
        if class_id <= len(self.classDataID):
            return self.classDataID[class_id]
        else:
            print 'Given class_id not exist, class_id:', self.classDataID.keys()
        
    def class_name2class_id(self, class_name):
        
        class_name = class_name.lower()
        if len(self.classDataName) == 0:
            self.loadClassData()
            
        try:
            return self.classDataName[class_name] 
#             return id
        except:
            print 'Given class_name not exist, class_name:', self.classDataName.keys()

    
#     def loadClassData(self):
#         # Load from DB
#         if self.db_connected:
#             cursor = self.db.cursor()
#             
#             sql = """SELECT *
#                      FROM Class
#                      """ 
#             cursor.execute(sql)
#             for c in cursor.fetchall():
#                 self.classDataID[c[0]] = c[1]
#         # Load from local data
#         elif self.localDataPath is not None: 
# 
#             path = os.path.join(self.localDataPath, 'class.pkl')
#             with open(path, 'rb') as infile:       
#                 self.classDataID = pickle.load(infile)
#         else:
#             print 'No data source found.'
#         self.classDataName = {y:x for x,y in self.classDataID.iteritems()}
    
    # File format: paper_id_image_id.{png, jpg,....}
    @ staticmethod
    def getIDsFromFilename(filename):
        filename = filename.split('.')
        filename = filename[0].split('_')
        return filename[1], filename[3]
    
    
    
    
    

if __name__ == '__main__': 
    
#     print ImageDataManager.getIDsFromFilename('paper_49_image_31.jpg')
#     
    DBInfoPath = '/Users/sephon/Desktop/Research/VizioMetrics/db_info.txt'
    db_info = ImageDataManager.getDBInfoFromFile(DBInfoPath)
    
    print db_info
    query = "select img_loc from keys_s3 WHERE key_size > 10000000 AND img_format = 'jpg'"
    IDM = ImageDataManager(connectToDB = True, db_info = db_info)
    print IDM.getKeynamesByQuery(query)
        
#     host = 'ec2-54-175-101-195.compute-1.amazonaws.com'
#     username = 'master'
#     password = 'password'
#     dbname = 'VizioMetrics'
#          
#          
#     db = MySQLdb.connect(host, username, password, dbname)
#     cursor = db.cursor()
# #     sql = "SELECT * FROM image_info LIMIT 20"
#     sql = "select * from keys_s3 WHERE key_size > 10000000 AND img_format = 'jpg'"
# 
#     print cursor.execute(sql)
# #     print cursor.fetchone()
# #     
#     data = cursor.fetchall()
#     print data
#     cursor.execute("SELECT * FROM ImageFileSource")
#     print  cursor.fetchone()
#     print  cursor.fetchone()
#     print  cursor.fetchone()

#     test = 'test'
#     sql += 'dddd%s' % test
#     print sql
#     print IDM.getKeynames(['jpg', 'tif'], 200, 123)
        