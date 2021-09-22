import os
import glob


def get_cutoff_index(cutoff):
    for i in range(len(smap_list)):
        if os.path.getsize(smap_list[i]) > cutoff:
            return i


path_root = path_root = '/cephfs_ryan/Batch_2/'

smap_list = glob.glob(path_root + 'Registered_SMAP_Data/*HEGOUT.tif')
smap_list.sort(key=os.path.getsize)
sar_list = glob.glob(path_root + 'Registered_SAR_Data/*VV*.tif')

cutoff = 60e3
smap_to_filter = smap_list[:get_cutoff_index(cutoff)]
sar_to_filter = []
for smap_file in smap_to_filter:
    for sar_file in sar_list:
        if smap_file.split('/')[-1].split('_')[6] == sar_file.split('/')[-1].split('-')[4]:
            sar_to_filter.append(sar_file)
            break

sar_dest = '/cephfs_ryan/Batch_2/Filtered_Out_SAR_Data/'
smap_dest = '/cephfs_ryan/Batch_2/Filtered_Out_SMAP_Data/'
for sar, smap in zip(sar_to_filter, smap_to_filter):
    os.system(f'mv {sar} {sar_dest}')
    os.system(f'mv {smap} {smap_dest}')

print('Program Terminated')
