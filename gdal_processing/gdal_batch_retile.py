import os
import glob
from add_config_capability import add_configs


config, config_file_name = add_configs()
sar_list = sorted(glob.glob(config.path_root + 'Cropped_SAR_Data/Window_' + str(config.win_size) + '_Overlap_' +
                            str(int(config.overlap * 100)) + '/*.tif'))
smap_list = sorted(glob.glob(config.path_root + 'Cropped_SMAP_Data/Window_' + str(config.win_size) + '_Overlap_' +
                             str(int(config.overlap * 100)) + '/*HEGOUT.tif'))

for sar_file, smap_file in zip(sar_list, smap_list):
    new_sar_dir = config.path_root + 'Retiled_SAR_Data/Window_' + str(config.win_size) + '_Overlap_' + \
                  str(int(config.overlap * 100)) + '/' + sar_file.split('/')[-1].split('.')[0]
    new_smap_dir = config.path_root + 'Retiled_SMAP_Data/Window_' + str(config.win_size) + '_Overlap_' + \
                   str(int(config.overlap * 100)) + '/' + smap_file.split('/')[-1].split('.')[0]
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

    bash_cmd = 'gdal_retile.py -ps ' + str(config.win_size) + ' ' + str(config.win_size) \
               + ' -overlap ' + str(int(config.win_size * config.overlap)) \
               + ' -targetDir ' + new_sar_dir + ' ' + sar_file
    os.system(bash_cmd)
    bash_cmd = 'gdal_retile.py -ps ' + str(config.win_size) + ' ' + str(config.win_size) \
               + ' -overlap ' + str(int(config.win_size * config.overlap)) \
               + ' -targetDir ' + new_smap_dir + ' ' + smap_file
    os.system(bash_cmd)

print("Program Terminated")
