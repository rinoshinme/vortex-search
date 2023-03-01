import os
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from image_search import SearchEngine
from image_search import get_config


def is_image_file(image_name):
    if '.' not in image_name:
        return False
    
    ext = image_name.split('.')[-1].lower()
    if ext in ['jpg', 'png', 'jpeg', 'bmp']:
        return True
    else:
        return False


def test_add(engine, image_folder):
    count = 0
    for root, dirs, files in os.walk(image_folder):
        for name in files:
            if not is_image_file(name):
                continue
            image_path = os.path.join(root, name)
            count += 1
            print('adding {}: {}'.format(count, image_path))
            engine.add_image(image_path)
            if count % 100 == 0:
                engine.save()
    engine.save()


def test_search(engine, image_path):
    results = engine.search_image(image_path, topk=2)
    print(results)
    for r in results:
        image = cv2.imread(r)
        image = resize_keepratio(image, 0.3)
        cv2.imshow('image', image)
        cv2.waitKey(0)


def resize_keepratio(image, scale=0.4):
    h, w = image.shape[:2]
    th = int(h * scale)
    tw = int(w * scale)
    return cv2.resize(image, (tw, th))


def main():
    config_path = 'configs/faiss_search.yaml'
    configs = get_config(config_path)

    engine = SearchEngine(configs)
    print(engine.count())

    # add
    # logo_root = 'D:/data/nsfw/data5_test/0normal'
    # test_add(engine, logo_root)

    # search
    image_path = 'D:/data/nsfw/data5_test/2sexy/0nJtAtm.jpg'
    test_search(engine, image_path)


if __name__ == '__main__':
    main()
