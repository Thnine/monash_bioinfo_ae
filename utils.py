import numpy as np
import pydicom
from PIL import Image


def dcm_to_jpg(dcm_file_path,jpg_file_path):


    ds = pydicom.dcmread(dcm_file_path)

    pixel_array = ds.pixel_array

    # 归一化像素值到0-255
    normalized_array = (pixel_array - np.min(pixel_array)) / (np.max(pixel_array) - np.min(pixel_array)) * 255
    normalized_array = normalized_array.astype(np.uint8)

    # 创建PIL图像对象
    image = Image.fromarray(normalized_array)

    # 保存为JPEG文件
    image.save(jpg_file_path)
