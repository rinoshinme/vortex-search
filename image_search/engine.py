import os
import numpy as np
import cv2
from .models import get_model
from .database import FaissDB, SqliteDB


class SearchEngine(object):
    def __init__(self, config):
        # TODO: use md5/rel_path instead of absolute paths
        self.config = config
        # self.feature_extractor = Resnet50()
        self.feature_extractor = get_model(config.MODEL.TYPE, config.MODEL.NAME, config.MODEL.WEIGHTS)
        self.faiss_db = FaissDB(config.FAISS.FEATURE_DIM, config.FAISS.INDEX_TYPE, config.FAISS.INDEX_PATH)
        self.sqlite_db = SqliteDB(config.SQLITE.DB_PATH, config.SQLITE.TABLE_NAME)
        self.min_similairity = 0.3
    
    def add_image(self, image_path):
        if self.sqlite_db.exists(image_path):
            # skip this image if it is already added.
            return
        
        # extract feature
        feat = self.feature_extractor.extract(image_path)
        # insert into database
        faiss_id = self.faiss_db.insert(feat)
        self.sqlite_db.insert(faiss_id, image_path)
    
    def exists(self, image_path):
        return self.sqlite_db.exists(image_path)
    
    def search_by_image(self, image_path, topk=1):
        # image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        feature = self.feature_extractor.extract(image_path)

        # search
        distances, indices = self.faiss_db.search(feature, topk)
        result_paths = []
        for index in indices[0]:
            p = self.sqlite_db.search(index)
            result_paths.append(p)
        return result_paths
    
    def search_by_text(self, text, topK=1):
        pass
    
    def save(self):
        self.faiss_db.save_index(self.config.FAISS.INDEX_PATH)
        # self.sqlite_db.close()

    def count(self):
        n1 = self.faiss_db.count()
        n2 = self.sqlite_db.count()
        assert (n1 == n2)
        return n1
