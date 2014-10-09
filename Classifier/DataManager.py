import MySQLdb
import json
import pickle
class ImageDataManager:
    
    db = None
    db_connected = False
    localDataPath = None
    classDataID = {}
    classDataName = {}
    
    def __init__(self, localDataPath = None, connectToDB = False, db_info = None):
        
        if connectToDB & (db_info is not None):
            host = db_info['host']
            db_username = db_info['db_username']
            db_password = db_info['db_password']
            db_name = db_info['db_name']
            self.loginDB(host, db_username, db_password, db_name)
            self.db_connected = True
        elif localDataPath is not None:
            self.localDataPath = localDataPath
        else:
            print 'Warning! Don\'t find any data support!'
            
    # login mysql
    def loginDB(self, host, db_username, db_password, db_name):
        self.db = MySQLdb.connect("localhost","sephon","19831122","VizioMatrics" )
    
    
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

    
    def loadClassData(self):
        # Load from DB
        if self.db_connected:
            cursor = self.db.cursor()
            
            sql = """SELECT *
                     FROM Class
                     """ 
            cursor.execute(sql)
            for c in cursor.fetchall():
                self.classDataID[c[0]] = c[1]
        # Load from local data
        elif self.localDataPath is not None: 
            path = self.extendPath(self.localDataPath, 'class.pkl')
            with open(path, 'rb') as infile:       
                self.classDataID = pickle.load(infile)
        else:
            print 'No data source found.'
        self.classDataName = {y:x for x,y in self.classDataID.iteritems()}
    
    # Read image and update image_path by image_id
    @ staticmethod
    def updateImagePath():
        return
    
    # Read image from categorized directory and update isGroundTruth by image_id
    @ staticmethod
    def updateGroundTruthImageByCatDir():
        return
    
    # Update ground-truth class_id by image_id
    @ staticmethod
    def updateGroundTruthImageClassByCarDir(image_id, class_id):
        return
    
    # Update ground-truth subclass_id by image_id
    @ staticmethod
    def updateGroundTruthSubimageClassByCarDir(image_id, subclass_id):
        return
    
    
    @staticmethod
    def updateClfImageClass(image_id, class_id):
        return
    
    # Update ground-truth subclass_id by image_id
    @staticmethod
    def updateClfSubimageClass(image_id, class_id):
        return
    
    # File format: paper_id_image_id.{png, jpg,....}
    @ staticmethod
    def getIDsFromFilename(filename):
        filename = filename.split('.')
        filename = filename[0].split('_')
        return filename[1], filename[3]
    
    

if __name__ == '__main__': 
    
    print ImageDataManager.getIDsFromFilename('paper_49_image_31.jpg')
    
    dataPath = '/Users/sephon/Desktop/Research/VizioMetrics/DB/'
    db_info = { 'host': 'localhost',
                'db_username': 'sephon',
                'db_password': 19831122,
                'db_name': 'VizioMatrics'}
    
#     print db_info
    IDM = ImageDataManager(localDataPath = dataPath, connectToDB = True, db_info = db_info)

        
    
        
        
    db = MySQLdb.connect("localhost","sephon","19831122","VizioMatrics" )
    cursor = db.cursor()
    sql = """SELECT *FROM Class""" 
#     print cursor.execute(sql)
    
#     data = cursor.fetchall()
#     cursor.execute("SELECT * FROM ImageFileSource")
#     print  cursor.fetchone()
#     print  cursor.fetchone()
#     print  cursor.fetchone()


        
        