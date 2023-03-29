import os
import numpy as np
import cv2
from detector import LogoDetector
from model import ArcfaceLogo

from db_utils import FaissDB, SqliteDB


FEATURE_DIM = 512
INDEX_TYPE = 'Flat'
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
INDEX_PATH = os.path.join(CURRENT_DIR, 'logo.index')
DB_PATH = os.path.join(CURRENT_DIR, 'logo.db')
TABLE_NAME = 'logo_det'
DISTANCE_THRESHOLD = 0.6


class SearchEngine(object):
    def __init__(self):
        self.feature_extractor = ArcfaceLogo()
        self.detector = LogoDetector()
        self.faiss_db = FaissDB(FEATURE_DIM, INDEX_TYPE, INDEX_PATH)
        self.sqlite_db = SqliteDB(DB_PATH, TABLE_NAME)
        self.max_distance = DISTANCE_THRESHOLD

    def upload(self, image_path, logo_name, box=None):
        if self.sqlite_db.exists(image_path):
            # skip this image if it is already added.
            return
        
        image = cv2.imread(image_path)
        if image is None:
            return None
        
        # extract feature
        if box is not None:
            xmin = box['xmin']
            xmax = box['xmax']
            ymin = box['ymin']
            ymax = box['ymax']
            image = image[ymin:ymax, xmin:xmax, :]

        feat = self.feature_extractor.extract(image)

        # insert into database
        faiss_id = self.faiss_db.insert(feat)
        self.sqlite_db.insert(faiss_id, image_path, logo_name)
        return faiss_id
    
    def exists(self, image_path):
        return self.sqlite_db.exists(image_path)
    
    def search(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)

        boxes = self.detector.detect(image)
        logo_info = []
        for box in boxes:
            x1, y1, x2, y2 = box
            image_clip = image[y1:y2, x1:x2, :]
            feature = self.feature_extractor.extract(image_clip)

            # search
            distances, indices = self.faiss_db.search(feature, topk=1)
            distance = distances[0]
            if distance > self.max_distance:
                continue

            name = self.sqlite_db.search(index)
            logo_info.append({
                'name': name, 
                'xmin': x1, 
                'ymin': y1,
                'xmax': x2,
                'ymax': y2,
            })
        return logo_info
    
    def save(self):
        self.faiss_db.save_index(INDEX_PATH)
        # self.sqlite_db.close()

    def count(self):
        n1 = self.faiss_db.count()
        n2 = self.sqlite_db.count()
        assert (n1 == n2)
        return n1
