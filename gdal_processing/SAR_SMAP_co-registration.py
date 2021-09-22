import os
from gdal_processing.gdalinfo_python_bindings import get_data
import numpy as np


def lat_lon_pixel(gt, xInd, yInd):
    x = gt[0] + xInd*gt[1] + yInd*gt[2]
    y = gt[3] + xInd*gt[4] + yInd*gt[5]
    return [x, y]


#
# Raw SAR Data Extraction
#
sar_path = "/home/ryan/Downloads/Dataset_Misc/S1A_IW_GRDH_1SDV_20180406T133350_20180406T133415_021347_024BD7_6B09.SAFE"\
           "/measurement/"
sar_file = "s1a-iw-grd-vv-20180406t133350-20180406t133415-021347-024bd7-001.tiff"
print(f"GDAL info {sar_file}\n")
os.system("gdalinfo " + sar_path + sar_file)
raw_data = get_data(sar_path, sar_file)
raw_img = raw_data.band.ReadAsArray()


#
# SAR Data Extraction
#
output_file = "test-gdal.tiff"
bash_cmd = "gdalwarp -t_srs EPSG:4326 -ot Float32 " + sar_path + sar_file + \
           " " + sar_path + output_file
os.system(bash_cmd)
print(f"\nGDAL info {output_file}\n")
os.system("gdalinfo " + sar_path + output_file)
sar_data = get_data(sar_path, output_file)
sar_img = sar_data.band.ReadAsArray()
print("\nSAR nonzero elements:\n", np.transpose(np.nonzero(sar_img[0:100])))


#
# SMAP Data Extraction
#
smap_path = "/home/ryan/Downloads/Dataset_Misc/SMAP_L2_SM_SP_1AIWDV_20180405T140447_20180406T133350_112W40N_R16010_001/"
smap_file = "SMAP_L2_SM_SP_1AIWDV_20180405T140447_20180406T133350_" \
            "112W40N_R16010_001_Soil_Moisture_Retrieval_Data_1km.tif"
print(f"\nGDAL info {smap_file}\n")
os.system("gdalinfo " + smap_path + smap_file)
smap_data = get_data(smap_path, smap_file)
smap_img = smap_data.band.ReadAsArray()
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
smap_img[smap_img == -9999] = 0  # May not need this
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
print("\nSMAP nonzero elements:\n", np.transpose(np.nonzero(smap_img)))


#
# Generate a SAR image with the same extents as the SMAP image.
#
te_flag = str(smap_data.geo_ext[1][0]) + " " + str(smap_data.geo_ext[1][1]) + " " + \
          str(smap_data.geo_ext[2][0]) + " " + str(smap_data.geo_ext[2][1])
output_file = "test-gdal-smap-extents.tiff"
bash_cmd = "gdalwarp -t_srs EPSG:4326 -te " + te_flag + " -ot Float32 " + \
           sar_path + sar_file + " " + sar_path + output_file
os.system(bash_cmd)
print(f"\nGDAL info {output_file}\n")
os.system("gdalinfo " + sar_path + output_file)
sar_extent_data = get_data(sar_path, output_file)
sar_extent_img = sar_extent_data.band.ReadAsArray()
print("\nSAR with SMAP extents nonzero elements:\n", np.transpose(np.nonzero(sar_extent_img[0:500])))


#
# Generate a SAR image with the same extents and resolution as the SMAP image using the default nearest neighbour
# resampling method.
#
output_file = "test-gdal-smap-extents+resolution-near.tiff"
bash_cmd = "gdalwarp -t_srs EPSG:4326 -te " + te_flag + " -ts " + str(smap_data.dataset.RasterXSize) + " " \
           + str(smap_data.dataset.RasterYSize) + " -ot Float32 "\
           + sar_path + sar_file + " " + sar_path + output_file
os.system(bash_cmd)


#
# Generate a SAR image with the same extents and resolution as the SMAP image using an averaging re-sampling method.
#
output_file = "test-gdal-smap-extents+resolution-average.tiff"
bash_cmd = "gdalwarp -t_srs EPSG:4326 -te " + te_flag + " -ts " + str(smap_data.dataset.RasterXSize) + " " \
           + str(smap_data.dataset.RasterYSize) + " -ot Float32 -r average " \
           + sar_path + sar_file + " " + sar_path + output_file
os.system(bash_cmd)
print(f"\nGDAL info {output_file}\n")
os.system("gdalinfo " + sar_path + output_file)
sar_low_res_data = get_data(sar_path, output_file)
sar_low_res_img = sar_low_res_data.band.ReadAsArray()
print("\nSAR Low Resolution nonzero elements:\n", np.transpose(np.nonzero(sar_low_res_img)))


print("Program Terminated")
