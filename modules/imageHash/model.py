"""

"""
import os
import cv2
import numpy as np


class PerceptualHash(object):
    def __init__(self):
        self.method = 'phash'
        self.dct_coefficients = None
        self.width = 8
        self.height = 8
    
    def extract(self, image, pack=True):
        if self.method == 'phash':
            return self.extract_phash(image, pack)
        return None
    
    def fit_image(self, image):
        if len(image.shape) == 3:
            if image.shape[-1] == 1:
                return image[:, :, 0]
            elif image.shape[-1] == 3:
                return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            elif image.shape[-1] == 4:
                return cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
            else:
                return None
        elif len(image.shape) == 2:
            return image
        else:
            return None

    def extract_phash(self, image, pack=True):
        if isinstance(image, str):
            image = cv2.imdecode(np.fromfile(image, dtype=np.uint8), -1)
        image = cv2.resize(image, (self.width, self.height))
        image = self.fit_image(image)

        coefs = self.get_dct_coefficients()
        dct = np.dot(np.dot(image, coefs), np.transpose(image))
        b = dct[:8][:8]
        average = np.mean(b)

        hash_vec = []
        for i in range(self.width):
            for j in range(self.height):
                if b[i, j] >= average:
                    hash_vec.append(1)
                else:
                    hash_vec.append(0)
        if pack:
            # faiss的二进制需要8bit打包成uint8，
            # 所以需要返回长度为8的uint8数组。
            hash_vec = self.pack_bits(hash_vec)
        return np.array(hash_vec, dtype=np.uint8)

    def pack_bits(self, vector):
        data = []
        length = len(vector) // 8
        for i in range(length):
            bits = vector[i*8:i*8+8]
            bit_value = sum([b*(2**(7-i)) for (i, b) in enumerate(bits)])
            data.append(bit_value)
        return data
    
    def get_dct_coefficients(self):
        if self.dct_coefficients is None:
            A = []
            for i in range(self.width):
                for j in range(self.height):
                    if i == 0: a = np.sqrt(1/8)
                    else: a = np.sqrt(2/8)
                    A.append(a * np.cos(np.pi * (2*j+1)*i / (2*8)))
            A = np.array(A)
            self.dct_coefficients = np.reshape(A, (self.width, self.height))
        return self.dct_coefficients

    def hamming_distance(self, vec1, vec2):
        d = 0
        length = vec1.shape[0]
        for i in range(length):
            if vec1[i] != vec2[i]:
                d += 1
        return d / length


if __name__ == '__main__':
    gen = PerceptualHash()

    # img1 = 'F:/Data/2023020315192258355781686_image.jpg'
    # img2 = 'F:/Data/2023020315194226485462299_temp.jpg'
    # feat1 = gen.extract(img1, pack=False)
    # feat2 = gen.extract(img2, pack=False)
    # distance = gen.hamming_distance(feat1, feat2)
    # print(distance)

    img = 'E:/603c442e4e2e0a89400506.jpg'
    feat1 = gen.extract(img, pack=False)
    print(feat1)
    feat2 = gen.extract(img, pack=True)
    print(feat2)
