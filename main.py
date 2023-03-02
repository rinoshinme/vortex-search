import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from image_search import SearchEngine
from image_search import get_config
from tools.naive_search import naive_search


def is_image_file(image_name):
    if '.' not in image_name:
        return False
    
    ext = image_name.split('.')[-1].lower()
    if ext in ['jpg', 'png', 'jpeg', 'bmp']:
        return True
    else:
        return False


def test_add(engine, image_folder, max_count=1000):
    count = 0
    for root, dirs, files in os.walk(image_folder):
        for name in files:
            if not is_image_file(name):
                continue
            image_path = os.path.join(root, name)
            count += 1
            if count > max_count:
                return
            print('adding {}: {}'.format(count, image_path))
            engine.add_image(image_path)
            if count % 100 == 0:
                engine.save()
    engine.save()


def test_search(engine, image_path):
    results = engine.search_image(image_path, topk=3)
    print(results)
    for r in results:
        image = cv2.imread(r)
        image = resize_keepratio(image, 1.0)
        cv2.imshow('image', image)
        cv2.waitKey(0)


def resize_keepratio(image, scale=0.4):
    h, w = image.shape[:2]
    th = int(h * scale)
    tw = int(w * scale)
    return cv2.resize(image, (tw, th))


if __name__ == '__main__':
    config_path = 'configs/resnet.yaml'
    configs = get_config(config_path)
    engine = SearchEngine(configs)

    image_folder = 'D:/data/image_search/00'
    test_add(engine, image_folder)
    # image_path = 'data/liuyifei.png'
    image_path = 'data/sample.jpeg'
    test_search(engine, image_path)

    # naive_search('D:/data/image_search/00', './data/liuyifei.png')
