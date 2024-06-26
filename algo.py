import os
import glob
import torch
import pydicom

import numpy as np

import torch.nn as nn
import torchvision.transforms

import torchvision
import torchvision.transforms as transforms

from network import MIL_Attention_FC


def normalize_image(img, pixel_value):
    # pixel_value = img.pixel_array
    # 将像素值转换为Hu值。ds .RescaleSlope表示斜率，这里读取的为1。ds .RescaleIntercept表示截距，这里读取的为-1024
    Hu = pixel_value * int(img.RescaleSlope) + int(img.RescaleIntercept)

    # 设置窗宽和窗位，这里以肺部为例
    window_width = 300  # 400  # 窗宽
    window_level = 75  # 窗位
    minWindow = window_level - window_width * 0.5
    Hu = (Hu - minWindow) / window_width
    # 做一下归一化
    Hu[Hu > 1] = 1
    Hu[Hu < 0] = 0
    return Hu


def read_image(img_path, transformation=None):
    img = pydicom.dcmread(img_path)
    img = normalize_image(img, img.pixel_array)[None,]
    img = torch.Tensor(img)
    # print(img.size())
    if transformation is not None:
        img = transformation(img)
    return img


def calculate(id,clinical_data,CT_list):

    test_aug2 = torchvision.transforms.Compose([
        transforms.Resize([384, 384]),
        transforms.Normalize((0.485, 0.485, 0.485), (0.224, 0.224, 0.224)),
    ])

    # for model construction
    model = MIL_Attention_FC()

    model = nn.DataParallel(model)
    model.load_state_dict(torch.load('./weights/best_source.pth', map_location=torch.device('cpu')))
    model = model.eval()

    params = {}
    for name, param in model.named_parameters():
        params[name] = param.detach().cpu().numpy()

    # load CT images
    CT_images = []
    for CT_name in CT_list:
        img = read_image(f"./data/CT/{id}/{CT_name}.dcm", None)[None, :]
        CT_images.append(img)
    CT_images = torch.cat(CT_images, dim=1)
    CT_images = test_aug2(CT_images)

    gender = clinical_data['gender']  # male = 0, female = 1
    age = clinical_data['age']  # age
    history_of_diabetes = clinical_data['history_of_diabetes']  # without = 0, with = 1
    history_of_hypertension = clinical_data['history_of_hypertension']  # without = 0, with = 1
    smoking_history = clinical_data['smoking_history']  # without = 0, with = 1

    drinking_history = clinical_data['drinking_history']  # without = [1, 0, 0, 0]; sometimes = [0, 1, 0, 0]; frequently = [0, 0, 1, 0]; else = [0, 0, 0, 1]
    family_history_of_tumor = clinical_data['family_history_of_tumor']  # without = 0, with = 1
    pathological_stage = clinical_data['pathological_stage']  # I = [1, 0, 0, 0, 0], II = [0, 1, 0, 0, 0]; III = [0, 0, 1, 0, 0]; IV = [0, 0, 0, 1, 0]; else = [0, 0, 0, 0, 1]
    perineural_invasion = clinical_data['perineural_invasion']  # without = [1, 0, 0], with = [0, 1, 0], else = [0, 0, 1]
    pathological_type = clinical_data['pathological_type']  # well = [1, 0, 0, 0], mix = [0, 1, 0, 0], poor = [0, 0, 1, 0], else = [0, 0, 0, 1]
    position = clinical_data['position']  # RCC = [1, 0, 0, 0]; LCC = [0, 1, 0, 0], REC = [0, 0, 1, 0], else = [0, 0, 0, 1]
    white_blood_cell_count = clinical_data['white_blood_cell_count']  # 单位: *109/L
    red_blood_cell_count = clinical_data['red_blood_cell_count']  # 单位: *109/L
    hemoglobin = clinical_data['hemoglobin']  # 单位: g/L
    platelet_concentration = clinical_data['platelet_concentration']  # 单位: *109/L
    neutrophil_count = clinical_data['neutrophil_count']  # 单位: *109/L
    lymphocyte_count = clinical_data['lymphocyte_count']  # 单位: *109/L
    monocyte_count = clinical_data['monocyte_count']  # 单位: *109/L
    red_cell_volumn_distribution_width = clinical_data['red_cell_volumn_distribution_width']
    plateletcrit = clinical_data['plateletcrit']
    mean_platelet_volume = clinical_data['mean_platelet_volume']
    albumin = clinical_data['albumin']
    globulin = clinical_data['globulin']
    albumin_globulin_ratio = clinical_data['albumin_globulin_ratio']
    blood_glucose = clinical_data['blood_glucose']
    triglyceride = clinical_data['triglyceride']
    cholesterol = clinical_data['cholesterol']
    high_density_lipoprotein = clinical_data['high_density_lipoprotein']
    low_density_lipoprotein = clinical_data['low_density_lipoprotein']
    carcinoembryonic_antigen = clinical_data['carcinoembryonic_antigen']
    carcinoembryonic_antigen_199 = clinical_data['carcinoembryonic_antigen_199']
    carcinoembryonic_antigen_125 = clinical_data['carcinoembryonic_antigen_125']

    # for shaolun designing the web server
    clinical_data = np.array(gender + age + history_of_diabetes + history_of_hypertension + smoking_history + drinking_history + family_history_of_tumor + pathological_stage + perineural_invasion + pathological_type + position + white_blood_cell_count
                               + red_blood_cell_count + hemoglobin + platelet_concentration + neutrophil_count + lymphocyte_count + monocyte_count + red_cell_volumn_distribution_width + plateletcrit + mean_platelet_volume + albumin
                               + globulin + albumin_globulin_ratio + blood_glucose + triglyceride + cholesterol + high_density_lipoprotein + low_density_lipoprotein + carcinoembryonic_antigen + carcinoembryonic_antigen_199 + carcinoembryonic_antigen_125)

    # clinical_data = np.array([0, 64.0, 0, 1, 1, 0, 0, 0, 1, 0,   0, 1, 0, 0, 0,   0, 0, 1,   1, 0, 0, 0,   1, 0, 0, 0,  6.1, 4.19, 124.0, 153.0, 3.8, 1.9, 0.3, 14.4, 0.2, 11.3, 42.8, 20.8, 2.1, 5.03, 1.23, 4.42, 1.61, 2.46, 0.2, 2.29, 1.08])

    max = np.array([1.0000e+00, 8.9000e+01, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00, 1.0000e+00,
        1.0000e+00, 1.0000e+00, 6.6330e+01, 1.6600e+01, 2.5900e+02, 9.3400e+02,
        5.7500e+01, 2.4800e+01, 2.5000e+00, 4.7000e+01, 1.5400e+02, 8.9600e+01,
        5.4100e+01, 4.5700e+01, 2.5000e+00, 2.7200e+01, 5.8050e+01, 9.5800e+00,
        4.8300e+00, 7.0400e+00, 8.2788e+02, 3.1891e+03, 3.6835e+03])

    min = np.array([0.0000e+00, 1.8000e+01, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
        0.0000e+00, 0.0000e+00, 6.6000e-01, 6.6000e-01, 1.1000e+01, 2.7700e+00,
        1.8000e-01, 1.0000e-01, 1.0000e-01, 3.7400e+00, 2.0000e-02, 3.4000e+00,
        7.8000e+00, 1.4900e+01, 3.0000e-01, 2.3400e+00, 2.8000e-01, 6.6000e-01,
        2.2000e-01, 8.1000e-01, 1.0000e-02, 7.0000e-02, 2.0000e-02])

    clinical_data = (clinical_data - min) / (max - min)
    clinical_data = torch.Tensor(clinical_data[None, ])

    data = (CT_images, clinical_data)
    prediction = model(data)
    prediction = torch.sigmoid(prediction)

    return (prediction.item())


if __name__ == '__main__':
    calculate()

    


