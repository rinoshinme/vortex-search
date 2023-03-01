import numpy as np
import cv2
from .models import Resnet50
from .database import FaissDB, SqliteDB


class SearchEngine(object):
    def __init__(self, config):
        self.config = config
        self.feature_extractor = Resnet50()
        self.faiss_db = FaissDB(index_path=config.FAISS.INDEX_PATH)
        self.sqlite_db = SqliteDB(db_path=config.SQLITE.DB_PATH)
        self.min_similairity = 0.3
    
    def add_image(self, image_path):
        # extract feature
        feat = self.feature_extractor.extract(image_path)
        # insert into database
        faiss_id = self.faiss_db.insert(feat)
        self.sqlite_db.insert(faiss_id, image_path)
    
    def search_image(self, image_path, topk=1):
        # image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        feature = self.feature_extractor.extract(image_path)

        # search
        distances, indices = self.faiss_db.search(feature, topk)
        result_paths = []
        for index in indices[0]:
            p = self.sqlite_db.search(index)
            result_paths.append(p)
        return result_paths
    
    def save(self):
        self.faiss_db.save_index(self.config.FAISS.INDEX_PATH)
        # self.sqlite_db.close()

    def count(self):
        n1 = self.faiss_db.count()
        n2 = self.sqlite_db.count()
        assert (n1 == n2)
        return n1
