import os
import shutil
import zipfile
import pathlib


class DataHandler:

    @staticmethod
    def extract_files():
        data_dir = './data'
        if os.path.isdir(data_dir) is False or len(os.listdir(data_dir)) == 0:
            with zipfile.ZipFile('zipped/train.zip', 'r') as zip_ref:
                zip_ref.extractall()
            with zipfile.ZipFile('zipped/validation.zip', 'r') as zip_ref:
                zip_ref.extractall()

            destination = data_dir
            newpath = r'./data'
            if not os.path.exists(newpath):
                os.makedirs(newpath)

            shutil.move('./train', destination)
            shutil.move('./validation', destination)

    @staticmethod
    def calculate_num_images():
        file_ext = '*.jpg'
        data_dir_cartoons_train = pathlib.Path('./data/train/cartoons')
        data_dir_photos_train = pathlib.Path('./data/train/photos')

        data_dir_cartoons_val = pathlib.Path('./data/validation/cartoons')
        data_dir_photos_val = pathlib.Path('./data/validation/photos')

        return (len(list(data_dir_cartoons_train.glob(file_ext))) + len(list(data_dir_photos_train.glob(file_ext))) +
                len(list(data_dir_cartoons_val.glob(file_ext))) + len(list(data_dir_photos_val.glob(file_ext)))) / 2
