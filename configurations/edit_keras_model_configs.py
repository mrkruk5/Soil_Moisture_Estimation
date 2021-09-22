import os
import glob
import json
import pprint


def update_path_root(file_name, config_dict, add_win_overlap, add_subdir):
    if add_win_overlap is True:
        config_dict['path_root'] = map['a']['path_root']['3'] + \
                                   '_'.join([map['b']['win_size']['0'],
                                             map['c']['overlap']['0']]) + '/'
    elif add_subdir is True:
        tmp = config_dict['path_root'].split('/')[-2].split('_')
        new_subdir = '_'.join(tmp[:2]) + '/' + '_'.join(tmp[2:]) + '/'
        new_path_root = '/'.join(config_dict['path_root'].split('/')[:-2]) + '/' + new_subdir
        config_dict['path_root'] = new_path_root
    print(file_name)
    pprint.pprint(config)
    print()
    f.seek(0)
    f.write(json.dumps(config_dict, indent=4))
    f.truncate()


def add_batch_size_and_epochs(file_name, config_dict):
    config_dict['batch_size'] = 4
    config_dict['epochs'] = 100
    print(file_name)
    pprint.pprint(config)
    print()
    f.seek(0)
    f.write(json.dumps(config_dict, indent=4))
    f.truncate()


def add_patience(file_name, config_dict):
    config_dict['early_stopping_patience'] = 25
    config_dict['reduce_lr_patience'] = 10
    print(file_name)
    pprint.pprint(config)
    print()
    f.seek(0)
    f.write(json.dumps(config_dict, indent=4))
    f.truncate()


def rename_file(old_name):
    os.system(f'echo {old_name}')
    new_number = list(old_name.split('_')[-1].split('.')[0])
    new_number.insert(1, list(map['b']['win_size'].keys())[list(map['b']['win_size'].values()).index('Window_32')])
    new_number.insert(2, list(map['c']['overlap'].keys())[list(map['c']['overlap'].values()).index('Overlap_0')])
    new_name = '_'.join(['_'.join(old_name.split('_')[:2]), ''.join(new_number)]) + '.json'
    os.system(f'git mv {old_name} {new_name}')


config_list = sorted(glob.glob('keras_model_*.json'))

with open('config_naming_scheme.json', 'r') as master:
    map = json.load(master)

for file_name in config_list:
    with open(file_name, 'r+') as f:
        config = json.load(f)
        # add_batch_size_and_epochs(file_name, config)
        # add_patience(file_name, config)
        # update_path_root(file_name, config, False, True)
        # rename_file(file_name)

print('Program Terminated')
