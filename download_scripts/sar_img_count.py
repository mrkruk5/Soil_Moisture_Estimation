import csv
import glob

path_root = '/home/ryan/Downloads/ESA_batch_script_tests'
sar_list = sorted(glob.glob(path_root + '/*/products-list.csv'))

global_count = 0
for file in sar_list:
    local_count = 0
    with open(file) as f:
        reader = csv.reader(f)
        for row in reader:
            local_count += 1
            global_count += 1
        print(f'Area {file.split("/")[5]} has {local_count} images.')
print(f'The total number of images across all areas is {global_count}.')

print('Program Terminated')
