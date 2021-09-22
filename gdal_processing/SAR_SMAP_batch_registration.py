import os
import glob
import gdal_python_functions as gpf


def sort_by_sar_date(element):
    return element.split('/')[-1][:20], element.split('/')[-1].split('_')[6]


def good_extents(sar_corners, smap_corners):
    # Assumes that all coordinates are in the northern hemisphere.
    good, bad = 0, 0
    if sar_corners[0][0] < smap_corners[0][0] or sar_corners[0][1] > smap_corners[0][1]:
        print('Upper Left SAR Corner is not contained by the SMAP image.')
        bad += 1
    if sar_corners[3][0] > smap_corners[3][0] or sar_corners[3][1] < smap_corners[3][1]:
        print('Lower Right SAR Corner is not contained by the SMAP image.')
        bad += 1
    print(f'SAR image is contained within the SMAP image? {good == bad}')
    return good == bad


path_root = '/cephfs_ryan/Batch_2/'
new_dir = path_root + 'Registered_SAR_Data/'
try:
    os.mkdir(new_dir)
    print('Directory: ', new_dir, ' Created.')
except FileExistsError:
    print('Directory: ', new_dir, ' already exists.')

sar_list = sorted(glob.glob(path_root + 'Raw_SAR_Data_Geocoded/*VV*.tif'))
smap_list = glob.glob(path_root + 'Registered_SMAP_Data/*HEGOUT.tif')
smap_list.sort(key=sort_by_sar_date)

if len(sar_list) == len(smap_list):
    count = 0
    for i in range(len(sar_list)):
        sar_time = sar_list[i].split("/")[-1].split("-")[4]
        smap_time = smap_list[i].split("/")[-1].split("_")[6]
        if sar_time == smap_time:
            count += 1
        else:
            print(f'{i} {sar_list[i].split("/")[-1]} != {smap_list[i].split("/")[-1]}')
    print(f'SAR dataset is correctly mapped to the SMAP dataset? {len(sar_list) == count}')
else:
    print('The SAR dataset is not the same size as the SMAP dataset.')

bad_pairs = []
for sar_file, smap_file in zip(sar_list, smap_list):
    sar_data = gpf.get_data(sar_file)
    sar_extents = sar_data.geo_ext  # [Upper L, Lower L, Upper R, Lower R]
    smap_data = gpf.get_data(smap_file)
    smap_extents = smap_data.geo_ext
    if good_extents(sar_extents, smap_extents):
        te_flag = str(smap_extents[1][0]) + " " + str(smap_extents[1][1]) + " " + \
                  str(smap_extents[2][0]) + " " + str(smap_extents[2][1])
        output_file = sar_file.split('/')[-1].split('.')[0].upper() + '.tif'
        bash_cmd = "gdalwarp -t_srs EPSG:4326 -te " + te_flag + " -ts " + \
                   str(smap_data.dataset.RasterXSize) + " " + \
                   str(smap_data.dataset.RasterYSize) + " -ot Float32 -r average " + \
                   sar_file + " " + new_dir + output_file
        os.system(bash_cmd)
    else:
        bad_pairs.append(sar_file + ' ' + smap_file)

if len(bad_pairs) > 0:
    with open(path_root + 'bad_pairs.txt', 'w') as f:
        for e in bad_pairs:
            f.write(e + '\n')

print("Program Terminated")
