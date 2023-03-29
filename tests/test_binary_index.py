import faiss
import random
import numpy as np


def make_data(size):
    data = []
    for i in range(size):
        vec = np.random.randint(0, 256, 8, dtype=np.uint8)
        data.append(vec)
    return data


def test():
    index = faiss.IndexBinaryFlat(64)
    # index = faiss.IndexBinaryIVF(quantizer, 64, 32)
    data = make_data(16)
    for item in data:
        item = np.expand_dims(item, axis=0)
        index.add(item)
    print(index.ntotal)
    print(index)
    faiss.write_index(index, './test.index')


if __name__ == '__main__':
    test()
