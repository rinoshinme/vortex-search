import os
import numpy as np
import faiss        # save feature vectors
import sqlite3      # save logo information


class SqliteDB(object):
    """
    store faiss id and logo_name pairs
    """
    def __init__(self, db_path):
        self.table_name = 'FaissLogo'
        self.conn = sqlite3.connect(db_path)
        
        cursor = self.conn.cursor()
        # create table if not exists
        command = "create table if not exists {} ( \
            FaissID integer primary key autoincrement, \
            ImagePath varchar(1024) \
        )".format(self.table_name)

        cursor.execute(command)
        self.conn.commit()
    
    def close(self):
        self.conn.close()
    
    def insert(self, faiss_id, image_path):
        command = "insert into {} values \
            ({}, \'{}\')".format(
            self.table_name, faiss_id, image_path)
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
            return results[0][1]
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
                 index_param='Flat', 
                 metric='cos', 
                 index_path=None):
        self.feature_dim = feature_dim
        self.metric = self.get_metric(metric)
        self.index_param = index_param
        self.load_index(index_path)

    def insert(self, feature):
        if len(feature.shape) == 1:
            feature = np.expand_dims(feature, axis=0)
        self.index.add(feature)
        database_id = self.index.ntotal
        return database_id
    
    def search(self, feature, topK=1):
        if len(feature.shape) == 1:
            feature = np.expand_dims(feature, axis=0)
        distances, indices = self.index.search(feature, topK)
        return distances, indices
    
    def get_metric(self, measurement):
        metric_map = {
            'cos': faiss.METRIC_INNER_PRODUCT,
            'l1': faiss.METRIC_L1,
            'l2': faiss.METRIC_L2,
        }
        return metric_map[measurement]

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
            self.index = faiss.index_factory(self.feature_dim, self.index_param, self.metric)
        else:
            self.index = faiss.read_index(index_path)
    
    def save_index(self, index_path):
        faiss.write_index(self.index, index_path)
    
    def count(self):
        return self.index.ntotal
