import os
import glob
import cv2
import numpy as np
import math
from add_config_capability import add_configs


def sort_by_sar_date(element):
    return element.split('/')[-1][:20], element.split('/')[-1].split('_')[6]


config, config_file_name = add_configs()
new_sar_dir = config.path_root + 'Cropped_SAR_Data/Window_' + str(config.win_size) + '_Overlap_' + \
              str(int(config.overlap * 100)) + '/'
new_smap_dir = config.path_root + 'Cropped_SMAP_Data/Window_' + str(config.win_size) + '_Overlap_' + \
              str(int(config.overlap * 100)) + '/'

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

for sar_file, smap_file in zip(sar_list, smap_list):
    smap_img = cv2.imread(smap_file, cv2.IMREAD_UNCHANGED)
    smap_shape = smap_img.shape
    valid_pixels = np.where(smap_img > 0)

    # TODO: Consider Hough Transform for line detection so I can distinguish where the image pixels are from the border.
    base_x_offset = np.min(valid_pixels[1])
    base_y_offset = np.min(valid_pixels[0])
    base_width = np.max(valid_pixels[1]) - np.min(valid_pixels[1])
    base_height = np.max(valid_pixels[0]) - np.min(valid_pixels[0])

    if config.overlap == 0:
        col_tiles = math.floor((base_width - config.win_size) / config.win_size + 1)
        row_tiles = math.floor((base_height - config.win_size) / config.win_size + 1)
        crop_width = col_tiles * config.win_size
        crop_height = row_tiles * config.win_size
    else:
        col_tiles = math.floor((base_width - config.win_size) / (config.win_size * (1 - config.overlap)) + 1)
        row_tiles = math.floor((base_height - config.win_size) / (config.win_size * (1 - config.overlap)) + 1)
        crop_width = config.win_size + config.win_size * (1 - config.overlap) * (col_tiles - 1)
        crop_height = config.win_size + config.win_size * (1 - config.overlap) * (row_tiles - 1)

    x_offset = int(base_x_offset - ((crop_width - base_width) / 2))
    y_offset = int(base_y_offset - ((crop_height - base_height) / 2))
    bash_cmd = f'gdal_translate -srcwin {x_offset} {y_offset} {crop_width} {crop_height}' \
        f' {sar_file} {new_sar_dir + sar_file.split("/")[-1]}'
    os.system(bash_cmd)
    bash_cmd = f'gdal_translate -srcwin {x_offset} {y_offset} {crop_width} {crop_height}' \
        f' {smap_file} {new_smap_dir + smap_file.split("/")[-1]}'
    os.system(bash_cmd)

print("Program Terminated")
