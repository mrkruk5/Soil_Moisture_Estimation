def sort_by_sar_date(element):
    return element[:20], element.split('_')[6]


path_root = '/home/ryan/git/presales/ryan_work/download_scripts/'

sar_files = []
with open(path_root + 'sar_file_list_filtered.txt', 'r') as f:
    for line in f:
        sar_files.append(line.strip('\n').split('/')[-1])
sar_files.sort()

with open(path_root + 'smap_file_list.txt', 'r') as f:
    smap_files = f.read().splitlines()
smap_files.sort(key=sort_by_sar_date)

for i in range(len(sar_files)):
    sar_time = sar_files[i].split("_")[4]
    smap_time = smap_files[i].split("_")[6]
    print(f'{i} {sar_files[i]}, {smap_files[i]}, Same timestamp? {sar_time == smap_time}')

print('Program Terminated')
