"""
create basic database for application
"""
import os
from engine import SearchEngine


def is_image_file(file_path):
    ext = file_path.split('.')[-1]
    if ext.lower in ['jpg', 'jpeg', 'bmp', 'png']:
        return True
    else:
        return False


def create(image_folder, max_count=-1):
    engine = SearchEngine()

    count = 0
    for root, dirs, files in os.walk(image_folder):
        for filename in files:
            count += 1
            if max_count > 0 and count > max_count:
                engine.save()
                engine.close()
                return
            
            filepath = os.path.join(root, filename)
            filepath = os.path.abspath(filepath)

            print(f'adding {filepath}')
            engine.upload(filepath)
    engine.save()


if __name__ == '__main__':
    image_folder = 'D:/data/image_search'
    create(image_folder, 100)
    