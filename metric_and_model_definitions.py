import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.python.keras import models
from tensorflow.python.keras import layers
from constants import config, smap_fill_value_scaled
from pre_processing import var_map


def masked_ae(y_true, y_pred):
    mask = y_true == -9999
    return K.abs(y_pred[~mask] - y_true[~mask])


def masked_ae_sum(y_true, y_pred):
    mask = y_true == -9999
    return K.sum(K.abs(y_pred[~mask] - y_true[~mask]))


def masked_mae(y_true, y_pred):
    mask = y_true == -9999
    return K.mean(K.abs(y_pred[~mask] - y_true[~mask]))


def masked_mse(y_true, y_pred):
    mask = tf.equal(y_true, smap_fill_value_scaled)
    masked_true = tf.boolean_mask(y_true, ~mask)
    masked_pred = tf.boolean_mask(y_pred, ~mask)
    diff = tf.subtract(masked_pred, masked_true)
    mse = K.mean(K.square(diff))
    return mse


def masked_normed_mse(y_true, y_pred):
    mask = tf.equal(y_true, smap_fill_value_scaled)
    masked_true = tf.boolean_mask(y_true, ~mask)
    masked_pred = tf.boolean_mask(y_pred, ~mask)
    diff = tf.subtract(masked_pred, masked_true)
    normed_mse = K.mean(K.square(diff)) / K.sqrt(K.sum(K.square(masked_true)))
    return normed_mse


def normed_mse(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true)) / K.sqrt(K.sum(K.square(y_true)))


def relative_L2_norm_squared(y_true, y_pred):
    return K.sum(K.square(y_pred - y_true)) / K.sum(K.square(y_true))


def r_squared(y_true, y_pred):
    shape = tf.shape(y_true)
    if 'flattened' in config.model:
        n = shape[0] * shape[1]
    else:
        n = shape[0] * shape[1] * shape[2]
    n = tf.to_float(n)
    a1 = (n*K.sum(y_true*y_pred) - K.sum(y_true)*K.sum(y_pred)) / \
         (n*K.sum(K.square(y_true)) - K.square(K.sum(y_true)) + K.epsilon())
    a0 = K.mean(y_pred) - a1*K.mean(y_true)
    y_line = a0 + a1*y_true
    y_mean = K.mean(y_pred)
    ss_tot = K.sum(K.square(y_pred - y_mean))
    ss_res = K.sum(K.square(y_pred - y_line))
    result = 1 - (ss_res / (ss_tot + K.epsilon()))
    return result


def standard_error_of_estimate(y_true, y_pred):
    shape = tf.shape(y_true)
    if 'flattened' in config.model:
        n = shape[0] * shape[1]
    else:
        n = shape[0] * shape[1] * shape[2]
    n = tf.to_float(n)
    return K.sqrt(K.sum(K.square(y_true - y_pred)) / (n - 2))


def base_model():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same',
                            data_format='channels_last', input_shape=var_map[config.input_shape]))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
    return model


def base_model_with_normalization():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same',
                            data_format='channels_last', input_shape=var_map[config.input_shape]))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
    return model


def deep_model():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same',
                            data_format='channels_last', input_shape=var_map[config.input_shape]))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(layers.BatchNormalization())
    model.add(layers.Activation('relu'))
    model.add(layers.Dense(units=32, activation='relu'))
    model.add(layers.BatchNormalization())
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
    return model


def dense_model():
    model = models.Sequential()
    model.add(layers.Dense(units=config.L1_units, activation='relu', input_shape=var_map[config.input_shape]))
    model.add(layers.Dense(units=config.L2_units, activation='relu'))
    model.add(layers.Dense(units=config.L3_units, activation='relu'))
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
    return model


def dense_flattened_model():
    model = models.Sequential()
    model.add(layers.Dense(units=config.L1_units, activation='sigmoid', input_shape=var_map[config.input_shape]))
    model.add(layers.Dense(units=config.L2_units, activation='sigmoid'))
    model.add(layers.Dense(units=config.L3_units, activation='sigmoid'))
    return model


def unet_like():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same',
                            data_format='channels_last', input_shape=var_map[config.input_shape]))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
    return model


def unet_like_with_regularization():
    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), strides=1, padding='same', data_format='channels_last',
                            input_shape=var_map[config.input_shape]))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), strides=1, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.MaxPooling2D(pool_size=(2, 2)))
    model.add(layers.Conv2DTranspose(filters=64, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Dropout(rate=0.1))
    model.add(layers.Conv2DTranspose(filters=32, kernel_size=(3, 3), strides=2, padding='same'))
    model.add(layers.Activation('relu'))
    model.add(layers.Conv2D(filters=1, kernel_size=(1, 1)))
    return model


def unet_mini():
    num_filters = 64
    input = layers.Input(var_map[config.input_shape])

    c1 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(input)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)

    c2 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(p1)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)

    c3 = layers.Conv2D(filters=num_filters*4, kernel_size=(3, 3), strides=1, padding='same')(p2)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(filters=num_filters*4, kernel_size=(3, 3), strides=1, padding='same')(c3)
    c3 = layers.Activation('relu')(c3)

    u4 = layers.Conv2DTranspose(filters=num_filters*2, kernel_size=(3, 3), strides=2, padding='same')(c3)
    u4 = layers.merge.concatenate([u4, c2])
    c4 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(u4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(c4)
    c4 = layers.Activation('relu')(c4)

    u5 = layers.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), strides=2, padding='same')(c4)
    u5 = layers.merge.concatenate([u5, c1])
    c5 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(u5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(c5)
    c5 = layers.Activation('relu')(c5)

    output = layers.Conv2D(filters=1, kernel_size=(1, 1))(c5)
    output = layers.Activation('linear')(output)
    model = models.Model(inputs=input, outputs=output)
    return model


def unet_mini_with_regularization():
    num_filters = 32
    dropout_prob = 0
    input = layers.Input(var_map[config.input_shape])

    c1 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(input)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    c1 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(c1)
    c1 = layers.BatchNormalization()(c1)
    c1 = layers.Activation('relu')(c1)
    p1 = layers.MaxPooling2D(pool_size=(2, 2))(c1)
    p1 = layers.Dropout(rate=dropout_prob)(p1)

    c2 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(p1)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    c2 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(c2)
    c2 = layers.BatchNormalization()(c2)
    c2 = layers.Activation('relu')(c2)
    p2 = layers.MaxPooling2D(pool_size=(2, 2))(c2)
    p2 = layers.Dropout(rate=dropout_prob)(p2)

    c3 = layers.Conv2D(filters=num_filters*4, kernel_size=(3, 3), strides=1, padding='same')(p2)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)
    c3 = layers.Conv2D(filters=num_filters*4, kernel_size=(3, 3), strides=1, padding='same')(c3)
    c3 = layers.BatchNormalization()(c3)
    c3 = layers.Activation('relu')(c3)

    u4 = layers.Conv2DTranspose(filters=num_filters*2, kernel_size=(3, 3), strides=2, padding='same')(c3)
    u4 = layers.merge.concatenate([u4, c2])
    u4 = layers.Dropout(rate=dropout_prob)(u4)
    c4 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(u4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)
    c4 = layers.Conv2D(filters=num_filters*2, kernel_size=(3, 3), strides=1, padding='same')(c4)
    c4 = layers.BatchNormalization()(c4)
    c4 = layers.Activation('relu')(c4)

    u5 = layers.Conv2DTranspose(filters=num_filters, kernel_size=(3, 3), strides=2, padding='same')(c4)
    u5 = layers.merge.concatenate([u5, c1])
    u5 = layers.Dropout(rate=dropout_prob)(u5)
    c5 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(u5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)
    c5 = layers.Conv2D(filters=num_filters, kernel_size=(3, 3), strides=1, padding='same')(c5)
    c5 = layers.BatchNormalization()(c5)
    c5 = layers.Activation('relu')(c5)

    output = layers.Conv2D(filters=1, kernel_size=(1, 1))(c5)
    model = models.Model(inputs=input, outputs=output)
    return model


def res_unet_mini():
    def conv_block(x, filters, kernel_size=(3, 3), strides=1, padding='same'):
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        conv = layers.Activation('relu')(conv)
        return conv

    def res_block(x, filters, kernel_size=(3, 3), strides=1, padding='same'):
        c1 = conv_block(x, filters, kernel_size=kernel_size, strides=strides, padding=padding)
        c2 = conv_block(c1, filters, kernel_size=kernel_size, strides=1)
        identity = layers.Conv2D(filters=filters, kernel_size=(1, 1), strides=strides, padding=padding)(x)
        output = layers.Add()([identity, c2])
        return output

    def deconv_block(x, skip, filters, kernel_size=(3, 3), strides=2, padding='same'):
        u1 = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        cat = layers.merge.concatenate([u1, skip])
        return cat

    num_filters = [64, 128, 256, 512]
    input = layers.Input(var_map[config.input_shape])
    e1 = res_block(input, num_filters[0], strides=1)
    e2 = res_block(e1, num_filters[1], strides=2)
    e3 = res_block(e2, num_filters[2], strides=2)

    b1 = conv_block(e3, num_filters[3], strides=1)
    b2 = conv_block(b1, num_filters[3], strides=1)

    u1 = deconv_block(b2, e2, num_filters[1])
    d1 = res_block(u1, num_filters[2])
    u2 = deconv_block(d1, e1, num_filters[0])
    d2 = res_block(u2, num_filters[1])

    output = layers.Conv2D(filters=1, kernel_size=(1, 1))(d2)
    model = models.Model(inputs=input, outputs=output)
    return model


def unet():
    # TODO: Make sure I give credit to
    #  https://github.com/nikhilroxtomar/UNet-Segmentation-in-Keras-TensorFlow/blob/master/unet-segmentation.ipynb
    #  for code.
    def down_block(x, filters, kernel_size=(3, 3), strides=1, padding='same'):
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(conv)
        conv = layers.Activation('relu')(conv)
        pool = layers.MaxPooling2D(pool_size=(2, 2))(conv)
        return conv, pool

    def up_block(x, skip, filters, kernel_size=(3, 3), strides=1, padding='same'):
        u1 = layers.Conv2DTranspose(filters=filters, kernel_size=kernel_size, strides=2, padding=padding)(x)
        cat = layers.merge.concatenate([u1, skip])
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(cat)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(conv)
        conv = layers.Activation('relu')(conv)
        return conv

    def bottleneck(x, filters, kernel_size=(3, 3), strides=1, padding='same'):
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(x)
        conv = layers.Activation('relu')(conv)
        conv = layers.Conv2D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding)(conv)
        conv = layers.Activation('relu')(conv)
        return conv

    num_filters = [32, 64, 128, 256, 512]
    input = layers.Input(var_map[config.input_shape])
    p0 = input
    c1, p1 = down_block(p0, num_filters[0])
    c2, p2 = down_block(p1, num_filters[1])
    c3, p3 = down_block(p2, num_filters[2])
    c4, p4 = down_block(p3, num_filters[3])

    bn = bottleneck(p4, num_filters[4])

    u1 = up_block(bn, c4, num_filters[3])
    u2 = up_block(u1, c3, num_filters[2])
    u3 = up_block(u2, c2, num_filters[1])
    u4 = up_block(u3, c1, num_filters[0])

    output = layers.Conv2D(filters=1, kernel_size=(1, 1))(u4)
    output = layers.Activation('linear')(output)
    model = models.Model(inputs=input, outputs=output)
    return model


func_map = {
    'mae': 'mae',
    'mse': 'mse',
    'masked_ae': masked_ae,
    'masked_ae_sum': masked_ae_sum,
    'masked_mae': masked_mae,
    'masked_mse': masked_mse,
    'masked_normed_mse': masked_normed_mse,
    'normed_mse': normed_mse,
    'relative_L2_norm_squared': relative_L2_norm_squared,
    'r_squared': r_squared,
    'standard_error_of_estimate': standard_error_of_estimate,
    'base_model': base_model,
    'deep_model': deep_model,
    'dense_model': dense_model,
    'dense_flattened_model': dense_flattened_model,
    'unet_like': unet_like,
    'unet_like_with_regularization': unet_like_with_regularization,
    'unet_mini': unet_mini,
    'unet_mini_with_regularization': unet_mini_with_regularization,
    'res_unet_mini': res_unet_mini,
    'unet': unet
}
