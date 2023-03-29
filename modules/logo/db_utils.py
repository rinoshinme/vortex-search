"""
database utility for this model
"""
import os
import numpy as np
import faiss        # save feature vectors
import sqlite3      # save logo information


"""
FaissID, ImagePath, LogoName
"""


class SqliteDB(object):
    """
    faiss id, image path
    """
    def __init__(self, db_path, table_name):
        self.table_name = table_name
        db_folder = os.path.split(db_path)[0]
        if not os.path.exists(db_folder):
            os.makedirs(db_folder)
        
        self.conn = sqlite3.connect(db_path)
        
        cursor = self.conn.cursor()
        # create table if not exists
        command = "create table if not exists {} ( \
            FaissID integer primary key autoincrement, \
            ImagePath varchar(1024), \
            LogoName varchar(1024) \
        )".format(self.table_name)

        cursor.execute(command)
        self.conn.commit()
    
    def close(self):
        self.conn.close()
    
    def exists(self, image_path):
        """
        check if a certain image exists in database
        """
        command = "select * from {} where ImagePath=\'{}\'".format(self.table_name, image_path)
        cursor = self.conn.cursor()
        cursor.execute(command)
        results = cursor.fetchall()
        if len(results) == 0:
            return False
        else:
            return True
    
    def insert(self, faiss_id, image_path, logo_name):
        command = "insert into {} values \
            ({}, \'{}\', \'{}\')".format(
            self.table_name, faiss_id, image_path, logo_name)
        cursor = self.conn.cursor()
        cursor.execute(command)
        self.conn.commit()
    
    def search(self, faiss_id):
        """
        search faiss id, return the corresponding image path
        """
        command = "select * from {} where FaissID={}".format(self.table_name, faiss_id)
        cursor = self.conn.cursor()
        cursor.execute(command)
        results = cursor.fetchall()
        if len(results) > 0:
            return results[0][2]  # return the name of the logo
        else:
            return None
    
    def count(self):
        command = "select count(FaissID) from {}".format(self.table_name)
        cursor = self.conn.cursor()
        cursor.execute(command)
        results = cursor.fetchall()
        return results[0][0]


class FaissDB(object):
    def __init__(self, 
                 feature_dim=2048, 
                 index_type='Flat', 
                 index_path=None):
        # arcface model feature
        self.feature_dim = feature_dim
        self.index_type = index_type

        index_folder = os.path.split(index_path)[0]
        if not os.path.exists(index_folder):
            os.makedirs(index_folder)
        self.load_index(index_path)

    def insert(self, feature):
        if len(feature.shape) == 1:
            feature = np.expand_dims(feature, axis=0)
        self.index.add(feature)
        database_id = self.index.ntotal
        return database_id - 1
    
    def search(self, feature, topK=1):
        if self.count() == 0:
            return [], []
            
        if len(feature.shape) == 1:
            feature = np.expand_dims(feature, axis=0)
        distances, indices = self.index.search(feature, topK)
        
        return distances[0], indices[0]

    def load_index(self, index_path=None):
        def is_valid(index_path):
            if index_path is None:
                return False
            elif not os.path.exists(index_path):
                return False
            elif not index_path.endswith('.index'):
                return False
            return True
        
        if not is_valid(index_path):
            if self.index_type == 'Flat':
                self.index = faiss.IndexFlatL2(self.feature_dim)
            else:
                raise ValueError("index type not supported yet")
        else:
            self.index = faiss.read_index(index_path)
    
    def save_index(self, index_path):
        faiss.write_index(self.index, index_path)
    
    def count(self):
        return self.index.ntotal
