import os
import numpy as np
import cv2
from clip import CLIPExtractor
from db_utils import FaissDB, SqliteDB


FEATURE_DIM = 512
INDEX_TYPE = 'Flat'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(CURRENT_DIR, 'clip.index')
DB_PATH = os.path.join(CURRENT_DIR, 'clip.db')
TABLE_NAME = 'clip'
DISTANCE_THRESHOLD = 0.3


class SearchEngine(object):
    def __init__(self):
        model_name = 'ViT-B/32'
        self.feature_extractor = CLIPExtractor(model_name)
        self.faiss_db = FaissDB(FEATURE_DIM, INDEX_TYPE, INDEX_PATH)
        self.sqlite_db = SqliteDB(DB_PATH, TABLE_NAME)
        self.min_similairity = DISTANCE_THRESHOLD

    def upload(self, image_path):
        if self.sqlite_db.exists(image_path):
            # skip this image if it is already added.
            return
        # extract feature
        feat = self.feature_extractor.extract(image_path)
        # insert into database
        faiss_id = self.faiss_db.insert(feat)
        self.sqlite_db.insert(faiss_id, image_path)
        return faiss_id
    
    def exists(self, image_path):
        return self.sqlite_db.exists(image_path)
    
    def search(self, text, topk=1):
        text_feature = self.feature_extractor.extract_text(text)
        distances, indices = self.faiss_db.search(text_feature, topk)
        result_paths = []
        for index in indices:
            p = self.sqlite_db.search(index)
            result_paths.append(p)
        return result_paths, distances.tolist()
    
    def save(self):
        self.faiss_db.save_index(INDEX_PATH)
        # self.sqlite_db.close()

    def count(self):
        n1 = self.faiss_db.count()
        n2 = self.sqlite_db.count()
        assert (n1 == n2)
        return n1
