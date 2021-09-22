import os
import glob
import cv2
import numpy as np
from add_config_capability import add_configs


def sort_by_sar_date(element):
    return element.split('/')[-1][:20], element.split('/')[-1].split('_')[6]


def find_dims():
    max_height, max_width = np.inf, np.inf
    for smap_file in smap_list:
        smap_img = cv2.imread(smap_file, cv2.IMREAD_UNCHANGED)
        height, width = smap_img.shape
        if height < max_height:
            max_height = height
        if width < max_width:
            max_width = width
    if max_height % 2 != 0:
        max_height -= 1
    if max_width % 2 != 0:
        max_width -= 1
    return max_height, max_width


config, config_file_name = add_configs()
new_sar_dir = config.path_root + 'Cropped_SAR_Data/Max_Crop_Size/'
new_smap_dir = config.path_root + 'Cropped_SMAP_Data/Max_Crop_Size/'

try:
    os.makedirs(new_sar_dir)
except FileExistsError:
    print('Directory:', new_sar_dir, 'already exists.')
else:
    print('Directory:', new_sar_dir, 'created.')

try:
    os.makedirs(new_smap_dir)
except FileExistsError:
    print('Directory:', new_smap_dir, 'already exists.')
else:
    print('Directory:', new_smap_dir, 'created.')

sar_list = sorted(glob.glob(config.path_root + 'Registered_SAR_Data/*.tif'))
smap_list = glob.glob(config.path_root + 'Registered_SMAP_Data/*HEGOUT.tif')
smap_list.sort(key=sort_by_sar_date)

max_height, max_width = find_dims()
print(f'Max height: {max_height}, Max width: {max_width}')

for sar_file, smap_file in zip(sar_list, smap_list):
    smap_img = cv2.imread(smap_file, cv2.IMREAD_UNCHANGED)
    height, width = smap_img.shape

    x_offset = int((width - max_width) / 2)
    y_offset = int((height - max_height) / 2)
    bash_cmd = f'gdal_translate -srcwin {x_offset} {y_offset} {max_width} {max_height} {sar_file} ' \
        f'{new_sar_dir + sar_file.split("/")[-1]}'
    os.system(bash_cmd)
    bash_cmd = f'gdal_translate -srcwin {x_offset} {y_offset} {max_width} {max_height} {smap_file} ' \
        f'{new_smap_dir + smap_file.split("/")[-1]}'
    os.system(bash_cmd)

print("Program Terminated")
