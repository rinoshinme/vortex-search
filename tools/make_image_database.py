import os
import hashlib
import shutil

DATABASE_ROOT = 'D:/data/image_search'
# DATABASE_ROOT = 'F:/data/image_search'


def get_file_md5(file_path):
    with open(file_path, 'rb') as f:
        data = f.read()
    m = hashlib.md5()
    m.update(data)
    return m.hexdigest()


def is_image_file(file_name):
    ext = file_name.split('.')[-1].lower()
    if ext in ['jpg', 'png', 'jpeg', 'bmp']:
        return True
    else:
        return False


def make_image_database(image_folder):
    print(image_folder)
    for root, dirs, files in os.walk(image_folder):
        for name in files:
            if not is_image_file(name):
                continue
            file_path = os.path.join(root, name)
            file_extension = name.split('.')[-1]
            md5_val = get_file_md5(file_path)
            print(md5_val)
            if len(md5_val) < 4:
                continue
            sub1 = md5_val[:2]
            sub2 = md5_val[2:4]
            save_name = f'{md5_val}.{file_extension}'
            save_folder = os.path.join(DATABASE_ROOT, sub1, sub2)
            if not os.path.exists(save_folder):
                os.makedirs(save_folder)
            save_path = os.path.join(save_folder, save_name)
            if not os.path.exists(save_path):
                shutil.move(file_path, save_path)


if __name__ == '__main__':
    make_image_database('D:/data/flickr30k/flickr30k-images')
