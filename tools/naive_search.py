import os
import numpy as np
from image_search.models import Resnet50 


def feature_similarity(feat1, feat2):
    return np.dot(feat1, feat2)


def naive_search(image_folder, ref_image_path):
    fe = Resnet50()
    features = []
    # add images
    count = 0
    for root, dirs, files in os.walk(image_folder):
        for name in files:
            image_path = os.path.join(root, name)
            print(image_path)
            feat = fe.extract(image_path)
            features.append((image_path, feat))
    
    # search
    ref_feat = fe.extract(ref_image_path)
    max_path = ''
    max_sim = 0.0
    for path, feat in features:
        sim = feature_similarity(feat, ref_feat)
        if sim > max_sim:
            max_sim = sim
            max_path = path
    
    print(max_sim)
    print(max_path)

