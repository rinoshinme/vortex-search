"""
test api by sending http requests
"""
import os
import json
import requests
import base64


class APITester(object):
    def __init__(self, base_url, port):
        self.base_url = base_url
        self.port = port
        self.headers = {"user-agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/75.0.3770.100 Safari/537.36",
               "Content-Type": 'application/json'}

    def hello(self):
        url = 'http://{}:{}/hello'.format(self.base_url, self.port)
        payload={}
        r = requests.post(url, json.dumps(payload), headers=self.headers)
        print(r.text)
    
    def upload(self):
        image_folder = 'D:/data/image_search/00/01/'
        url = 'http://{}:{}/img/upload'.format(self.base_url, self.port)
        for i in range(1, 10):
            image_path = os.path.join(image_folder, '{:06d}.jpg'.format(i))
            print('uploading {}'.format(image_path))
            image_data = self.file2base64(image_path)
            payload = {'image': image_data}
            r = requests.post(url, json.dumps(payload), headers=self.headers)
            print(r.text)

    def search(self):
        image_path = 'D:/data/image_search/00/01/00016f72e66c44190f2576d19fdb898a.jpg'
        url = 'http://{}:{}/img/search'.format(self.base_url, self.port)
        image_data = self.file2base64(image_path)
        payload = {'image': image_data}
        r = requests.post(url, json.dumps(payload), headers=self.headers)
        print(r.text)

    def file2base64(self, filepath):
        with open(filepath, 'rb') as f:
            data = f.read()
            data_b64 = base64.b64encode(data).decode()
        return data_b64


def test_api():
    tester = APITester('127.0.0.1', 1088)
    # tester.hello()
    # tester.upload()
    tester.search()


if __name__ == '__main__':
    test_api()