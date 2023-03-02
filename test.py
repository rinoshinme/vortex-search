import numpy as np
import time
from image_search.models import get_model


def test():
    model = get_model('clip', 'ViT-B/16')
    text = 'a taxi on the street'
    image_path = './data/sample.jpeg'
    st = time.time()
    tfeature = model.extract_text(text)
    ifeature = model.extract_image(image_path)
    et = time.time()
    print('inference time: ', et - st)

    sim = np.dot(tfeature, ifeature)
    print(sim)


if __name__ == '__main__':
    test()
