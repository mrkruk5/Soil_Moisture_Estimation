import os
import glob
import pickle
import tensorflow as tf
import rasterio
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from textwrap import wrap
from add_config_capability import add_configs


# Disable the warning: "Your CPU supports instructions that this TensorFlow binary was not compiled to use: AVX2 FMA"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


def vis_samples(num):
    iterator = dataset.make_initializable_iterator()
    next_set = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for i in range(num):
            sar_img, smap_img, sar_title, smap_title = sess.run(next_set)
            plot_image(sar_img, sar_title.decode('utf-8'), smap_img, smap_title.decode('utf-8'))


def plot_image(img1, name1, img2, name2):
    fig = plt.figure()
    plt.subplot(121)
    if config.exclude_samples_with_fill_values is False:
        masked_sar = np.ma.masked_where(img1 == 0, img1)
        cmap = mpl.cm.viridis
        cmap.set_bad(color='black')
        sar_img = plt.imshow(masked_sar, cmap=cmap)
    else:
        sar_img = plt.imshow(img1)
    plt.title('\n'.join(wrap('_'.join(name1.split('-')[:6]) + '_' + name1[-7:-4], 30)), fontsize=8)
    plt.colorbar(sar_img, fraction=0.05, pad=0.05)

    plt.subplot(122)
    if config.exclude_samples_with_fill_values is False:
        masked_smap = np.ma.masked_where(img2 == -9999, img2)
        cmap = mpl.cm.viridis
        cmap.set_bad(color='black')
        smap_img = plt.imshow(masked_smap, cmap=cmap)
    else:
        smap_img = plt.imshow(img2)
    plt.title('\n'.join(wrap(name2[:52] + name2[-8:-4], 37)), fontsize=8)
    plt.colorbar(smap_img, fraction=0.05, pad=0.05)
    plt.pause(0.1)
    plt.close(fig)


def open_image(path):
    with rasterio.open(path) as raster:
        img = raster.read(1)
    return img


def sort_by_sar_date(element):
    return element.split('/')[-1][:20], element.split('/')[-1].split('_')[6], \
           '_'.join(element.split('/')[-1].split('_')[-2:])


def create_dataset():
    sar_list = sorted(glob.glob(config.path_root + 'Retiled_SAR_Data/Window_' + str(config.win_size) + '_Overlap_' +
                                str(int(config.overlap * 100)) + '/*/*.tif'))
    smap_list = glob.glob(config.path_root + 'Retiled_SMAP_Data/Window_' + str(config.win_size) + '_Overlap_' +
                          str(int(config.overlap * 100)) + '/*/*.tif')
    smap_list.sort(key=sort_by_sar_date)

    # Feed the image set into a TensorFlow Dataset.
    dataset = tf.data.Dataset.from_generator(lambda: create_gen(sar_list, smap_list),
                                             output_types=(tf.float32, tf.float32, tf.string, tf.string))
    return dataset


def create_gen(sar_list, smap_list):
    sar_fill_value = 0
    smap_fill_value = -9999
    for sar_file, smap_file in zip(sar_list, smap_list):
        smap_img = open_image(smap_file)
        if config.exclude_samples_with_fill_values is True:
            if smap_fill_value not in smap_img:
                sar_img = open_image(sar_file)
                if sar_fill_value not in sar_img:
                    yield (sar_img, smap_img, sar_file.split('/')[-1], smap_file.split('/')[-1])
        else:
            sar_img = open_image(sar_file)
            yield (sar_img, smap_img, sar_file.split('/')[-1], smap_file.split('/')[-1])


def load_data_keras_format():
    width = height = config.win_size
    channels = 1
    dataset = create_dataset()

    sar_set, smap_set, sar_titles, smap_titles = [], [], [], []

    iterator = dataset.make_initializable_iterator()
    next_set = iterator.get_next()
    with tf.Session() as sess:
        sess.run(iterator.initializer)
        for sample in load_gen(sess, next_set):
            sar_img, smap_img, sar_title, smap_title = sample
            print(sar_title.decode('utf-8'), smap_title.decode('utf-8'))
            sar_set.append(sar_img.reshape((1, height, width, channels)))
            smap_set.append(smap_img.reshape((1, height, width, channels)))
            sar_titles.append(sar_title.decode('utf-8'))
            smap_titles.append(smap_title.decode('utf-8'))

    sar_set = np.concatenate(sar_set, axis=0)
    smap_set = np.concatenate(smap_set, axis=0)

    np.random.seed(0)
    np.random.shuffle(sar_set)
    np.random.seed(0)
    np.random.shuffle(smap_set)
    np.random.seed(0)
    np.random.shuffle(sar_titles)
    np.random.seed(0)
    np.random.shuffle(smap_titles)

    size_train = int(0.9 * sar_set.shape[0])
    size_test = sar_set.shape[0] - size_train

    sar_train = sar_set[:size_train, :]
    sar_train_titles = sar_titles[:size_train]
    sar_test = sar_set[-size_test:, :]
    sar_test_titles = sar_titles[-size_test:]

    smap_train = smap_set[:size_train, :]
    smap_train_titles = smap_titles[:size_train]
    smap_test = smap_set[-size_test:, :]
    smap_test_titles = smap_titles[-size_test:]
    return (sar_train, smap_train), (sar_test, smap_test),\
           (sar_train_titles, smap_train_titles), (sar_test_titles, smap_test_titles)


def load_gen(sess, next_set):
    while True:
        try:
            yield sess.run(next_set)
        except tf.errors.OutOfRangeError:
            break


if __name__ == '__main__':
    config, config_file_name = add_configs()

    #
    # Create a TensorFlow dataset.
    #
    dataset = create_dataset()

    #
    # Test that the images are being loaded in the TensorFlow Dataset and print the first 10 image pairs.
    #
    # vis_samples(10)

    #
    # Load the TensorFlow dataset into numpy arrays for Keras.
    #
    try:
        os.mkdir(config.new_dir)
    except FileExistsError:
        print('Directory:', config.new_dir, 'already exists.')
    else:
        print('Directory:', config.new_dir, 'created.')

    (sar_train, smap_train), (sar_test, smap_test),\
    (sar_train_titles, smap_train_titles), (sar_test_titles, smap_test_titles) = load_data_keras_format()

    with open(config.new_dir + 'sar_train.pickle', 'wb') as output_file:
        pickle.dump(sar_train, output_file)
    with open(config.new_dir + 'smap_train.pickle', 'wb') as output_file:
        pickle.dump(smap_train, output_file)
    with open(config.new_dir + 'sar_test.pickle', 'wb') as output_file:
        pickle.dump(sar_test, output_file)
    with open(config.new_dir + 'smap_test.pickle', 'wb') as output_file:
        pickle.dump(smap_test, output_file)
    with open(config.new_dir + 'sar_train_titles.pickle', 'wb') as output_file:
        pickle.dump(sar_train_titles, output_file)
    with open(config.new_dir + 'smap_train_titles.pickle', 'wb') as output_file:
        pickle.dump(smap_train_titles, output_file)
    with open(config.new_dir + 'sar_test_titles.pickle', 'wb') as output_file:
        pickle.dump(sar_test_titles, output_file)
    with open(config.new_dir + 'smap_test_titles.pickle', 'wb') as output_file:
        pickle.dump(smap_test_titles, output_file)

    print('Program Terminated')
