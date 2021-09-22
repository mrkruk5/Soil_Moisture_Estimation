import requests
from bs4 import BeautifulSoup as bs
import re
from calendar import monthrange


def search_page(link, regex):
    page = requests.get(link)
    page_html = bs(page.text, 'html.parser')
    result_set = page_html.find_all('a', string=regex)
    # TODO When I ran this file for all SAR files, I ran into some edge cases I didn't anticipate:
    #  1: There are sometimes two files that were generated under the same version for the same date and time,
    #     therefore a list is returned and nsidc-data-download.sh doesn't know how to handle this.
    #  2: Sometimes the regex will appear in both the SMAP date/time field as well as the SAR date/time field,
    #     therefore a list is returned and nsidc-data-download.sh doesn't know how to handle this.
    #  3: I manually re-ran nsidc-data-download.sh with a new smap_download_links.txt with the corrected 4 lines that
    #     contained the edge cases described.
    if len(result_set) > 1:
        link_list = []
        for tag in result_set:
            link_list.append(link + tag.string)
        return link_list
    elif len(result_set) == 1:
        for tag in result_set:
            link_file = link + tag.string
        return link_file
    else:
        return None


def find_dir(search_str):
    for link in link_dirs:
        if search_str in link:
            return link
    return None


def find_file(search_str):
    year = search_str.split('T')[0][:4]
    month = search_str.split('T')[0][4:6]
    day = search_str.split('T')[0][6:]
    cutoff = monthrange(int(year), int(month))[1]
    dir_str = f'{year}.{month}.{day}/'
    link_dir = find_dir(dir_str)
    link_file = search_page(link_dir, re.compile(search_str + '.*h5$'))
    if link_file is not None:
        print(f'Found {search_str} in {link_file}')
        return link_file

    # Check if the file is contained in the next day directory
    next_day = int(day) + 1
    if next_day > cutoff:
        next_month = int(month) + 1
        if next_month > 12:
            next_year = int(year) + 1
            next_month = 1
            next_day = 1
            next_dir_str = f'{next_year}.{str(next_month).zfill(len(month))}.{str(next_day).zfill(len(day))}/'
        else:
            next_dir_str = f'{year}.{str(next_month).zfill(len(month))}.{str(1).zfill(len(day))}/'
    else:
        next_dir_str = f'{year}.{month}.{str(next_day).zfill(len(day))}/'
    link_dir = find_dir(next_dir_str)
    link_file = search_page(link_dir, re.compile(search_str + '.*h5$'))
    if link_file is not None:
        print(f'Found {search_str} in {link_file}')
        return link_file

    # Check if the file is contained in the previous day directory
    prev_day = int(day) - 1
    if prev_day < 1:
        prev_month = int(month) - 1
        if prev_month < 1:
            prev_year = int(year) - 1
            prev_month = 12
            prev_day = monthrange(prev_year, prev_month)[1]
            prev_dir_str = f'{prev_year}.{prev_month}.{prev_day}/'
        else:
            prev_day = monthrange(int(year), prev_month)[1]
            prev_dir_str = f'{year}.{str(prev_month).zfill(len(month))}.{prev_day}/'
    else:
        prev_dir_str = f'{year}.{month}.{str(prev_day).zfill(len(day))}/'
    link_dir = find_dir(prev_dir_str)
    link_file = search_page(link_dir, re.compile(search_str + '.*h5$'))
    if link_file is not None:
        print(f'Found {search_str} in {link_file}')
        return link_file
    print(f'{search_str} was not found.')
    return search_str + ' was not found.'


link_root = 'https://n5eil01u.ecs.nsidc.org/SMAP/SPL2SMAP_S.002/'
link_dirs = search_page(link_root, re.compile('2018\.'))

with open('sar_file_list.txt') as file:
    sar_file_list = file.read().splitlines()


#
# Create a search list and remove duplicate timestamps.
#
filtered_sar_file_list = []
search_list = []
removed = []
for file in sar_file_list:
    search_str = file.split('/')[1].split('_')[4]
    if search_str not in search_list:
        search_list.append(search_str)
        filtered_sar_file_list.append(file)
    else:
        removed.append(file)
# list(dict.fromkeys(search_list)) also removes duplicates, but does not tell you which elements were duplicates.

with open('duplicate_sar_images.txt', 'w') as output:
    for e in removed:
        output.write(str(e) + '\n')


#
# Search for SMAP files.
#
link_files = []
misses = []
for i in range(len(search_list)):
    print(f'Searching for file {i}: {filtered_sar_file_list[i]} with search criteria {search_list[i]}')
    link_file = find_file(search_list[i])
    if 'was not found' in link_file:
        misses.append(filtered_sar_file_list[i])
    else:
        link_files.append(link_file)


#
# Write results to a set of files.
#
with open('unfound_sar_smap_images.txt', 'w') as output:
    for e in misses:
        output.write(str(e) + '\n')

with open('smap_download_links.txt', 'w') as output:
    for e in link_files:
        output.write(str(e) + '\n')

print('Program Terminated')
