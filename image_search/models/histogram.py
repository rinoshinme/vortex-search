import cv2
import numpy as np
from PIL import Image


class HistogramFeature(object):
    def __init__(self, image_size=128, grid_size=32):
        self.grid_size = grid_size
        self.image_size = image_size
        self.histogram_size = self.grid_size ** 3

    def get_histogram(self, image_path):
        image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), -1)
        
