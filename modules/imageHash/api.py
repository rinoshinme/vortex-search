import os
import platform
import uvicorn
import base64
import numpy as np
import cv2
import time
import hashlib
from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, List

from engine import SearchEngine


if platform.system() == 'Windows':
    TEMPORARY_IMAGE_FOLDER = 'D:/temp/image_search'
else:
    TEMPORARY_IMAGE_FOLDER = '/tmp/image_search'

if not os.path.exists(TEMPORARY_IMAGE_FOLDER):
    os.makedirs(TEMPORARY_IMAGE_FOLDER)


def file_md5(filepath):
	with open(filepath, 'rb') as f:
		data = f.read()
		md5_obj = hashlib.md5()
		md5_obj.update(data)
		code = md5_obj.hexdigest()
	return str(code).lower()


def data_md5(data):
    md5_obj = hashlib.md5()
    md5_obj.update(data)
    code = md5_obj.hexdigest()
    return str(code).lower()


def file_to_base64(filepath):
    with open(filepath, 'rb') as f:
        data = f.read()
        data_b64 = base64.b64encode(data).decode()
    return data_b64


class UploadParams(BaseModel):
    image: str = Field('image', description='image data', sqlalchemy_safe=True)
    timestamp: Optional[int] = Field(None, description="时间戳", sqlalchemy_safe=True)


class SearchParams(BaseModel):
    image: str = Field('image', description='image data', sqlalchemy_safe=True)
    topk: int = 1
    start_time: Optional[int] = Field(None, description="查询的起始时间戳", sqlalchemy_safe=True)
    end_time: Optional[int] = Field(None, description="查询的结束时间戳", sqlalchemy_safe=True)


ERROR_CODE_MAP = {
    4001: '请求参数错误',
    4002: '图像解析失败'
}


def get_error_message(code):
    if code == 0:
        return {'code': 0, 'message': 'success'}
    message = ERROR_CODE_MAP[code]
    return {'code': code, 'message': message}


app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


engine = SearchEngine()


@app.post('/hello')
async def hello():
    return {'code': 0, 'message': 'hello'}


@app.post('/img/upload')
async def upload(data: UploadParams):
    data = dict(data)

    if 'image' not in data.keys():
        return get_error_message(4001)
    try:
        image_data = data['image']
        image_data = bytes(image_data, encoding='utf-8')
        decoded = base64.urlsafe_b64decode(image_data)
        decoded = np.fromstring(decoded, np.uint8)
        image = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
    except Exception as e:
        return get_error_message(4002)
    
    # save image into local disk drive
    image_path = os.path.join(TEMPORARY_IMAGE_FOLDER, '{}.jpg'.format(data_md5(image_data)))
    if not os.path.exists(image_path):
        cv2.imwrite(image_path, image)

    index = engine.upload(image_path)
    return {'code': 0, 'message': {"id": index}}


@app.post('/img/search')
async def search(data: SearchParams):
    data = dict(data)
    if 'image' not in data.keys():
        return get_error_message(4001)
    if 'topk' not in data.keys():
        data['topk'] = 1
    
    # decode image data
    try:
        image_data = data['image']
        image_data = bytes(image_data, encoding='utf-8')
        decoded = base64.urlsafe_b64decode(image_data)
        decoded = np.fromstring(decoded, np.uint8)
        image = cv2.imdecode(decoded, cv2.IMREAD_COLOR)
    except Exception as e:
        return get_error_message(4002)
    
    image_paths, distances = engine.search(image, topk=data['topk'])

    return {'code': 0, 'message': {'image_paths': image_paths, 'distances': distances}}


if __name__ == '__main__':
    uvicorn.run(app=app, host='0.0.0.0', port=1088)
