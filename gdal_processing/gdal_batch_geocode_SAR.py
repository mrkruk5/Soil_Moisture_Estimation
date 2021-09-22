import os
import glob

path_root = '/cephfs_ryan/Batch_2/'
new_dir = path_root + 'Raw_SAR_Data_Geocoded/'
try:
    os.mkdir(new_dir)
    print('Directory: ', new_dir, ' Created.')
except FileExistsError:
    print('Directory: ', new_dir, ' already exists.')

sar_list = sorted(glob.glob(path_root + 'Raw_SAR_Data/*/*.SAFE/measurement/*vv*.tiff'))

for sar_file in sar_list:
    output_file = sar_file.split('/')[-1].split('.')[0].upper() + '.tif'
    bash_cmd = f'gdalwarp -t_srs EPSG:4326 -ot Float32 {sar_file} {new_dir + output_file}'
    os.system(bash_cmd)

print('Program Terminated')
