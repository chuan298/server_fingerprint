

from sklearn.model_selection import train_test_split

# path_img = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/image/"
# path_label = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/label/"


from PIL import Image
import numpy as np
import os, cv2

#import numpy as np
from tensorflow.keras.utils import Sequence
import cv2

import numpy as np
from tensorflow.keras.utils import Sequence
import cv2


class DataGenerator(Sequence):
    'Generates data for Keras'

    def __init__(self,
                 all_filenames,
                 labels,
                 batch_size,
                 input_dim,
                 n_channels,
                 normalize,
                 zoom_range,
                 rotation,
                 brightness_range,
                 shuffle=True):
        '''
        all_filenames: list toàn bộ các filename
        labels: nhãn của toàn bộ các file
        batch_size: kích thước của 1 batch
        input_dim: (width, height) đầu vào của ảnh
        n_channels: số lượng channels của ảnh
        normalize: Chuẩn hóa ảnh
        zoom_range: Kích thước scale
        rotation: Độ xoay của ảnh
        brightness_range: Độ sáng
        shuffle: có shuffle dữ liệu sau mỗi epoch hay không?
        '''
        self.all_filenames = all_filenames
        self.labels = labels
        self.batch_size = batch_size
        self.input_dim = input_dim
        self.n_channels = n_channels
        self.normalize = normalize
        self.zoom_range = zoom_range
        self.rotation = rotation
        self.brightness_range = brightness_range
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        '''
        return:
          Trả về số lượng batch/1 epoch
        '''
        return int(np.floor(len(self.all_filenames) / self.batch_size))

    def __getitem__(self, index):
        '''
        params:
          index: index của batch
        return:
          X, Y cho batch thứ index
        '''
        # Lấy ra indexes của batch thứ index
        indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]

        # List all_filenames trong một batch
        all_filenames_temp = [self.all_filenames[k] for k in indexes]

        # Khởi tạo data
        X, Y = self.__data_generation(all_filenames_temp)

        return X, Y

    def on_epoch_end(self):
        '''
        Shuffle dữ liệu khi epochs end hoặc start.
        '''
        self.indexes = np.arange(len(self.all_filenames))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, all_filenames_temp):
        '''
        params:
          all_filenames_temp: list các filenames trong 1 batch
        return:
          Trả về giá trị cho một batch.
        '''
        X = np.empty((self.batch_size, *self.input_dim, self.n_channels))
        Y = np.empty((self.batch_size, *self.input_dim, self.n_channels))

        # Khởi tạo dữ liệu
        for i, (fn, label_fn) in enumerate(all_filenames_temp):
            # Đọc file từ folder name
            img = cv2.imread(fn)
            label = cv2.imread(label_fn)
            # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.input_dim)
            label = cv2.resize(label, self.input_dim)

            if self.normalize:
                mean1 = np.mean(img, axis=0)
                std1 = np.std(img, axis=0)
                img = (img - mean1) / std1

            if self.zoom_range:
                zoom_scale = 1 / np.random.uniform(self.zoom_range[0], self.zoom_range[1])
                (h, w, c) = img.shape
                img = cv2.resize(img, (int(h * zoom_scale), int(w * zoom_scale)), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (int(h * zoom_scale), int(w * zoom_scale)), interpolation=cv2.INTER_LINEAR)
                label = label / 255
                label[label > 0.5] = 1
                label[label < 0.5] = 0
                (h_rz, w_rz, c) = img.shape
                start_w = np.random.randint(0, w_rz - w) if (w_rz - w) > 0 else 0
                start_h = np.random.randint(0, h_rz - h) if (h_rz - h) > 0 else 0
                # print(start_w, start_h)
                img = img[start_h:(start_h + h), start_w:(start_w + w), :].copy()
                label = label[start_h:(start_h + h), start_w:(start_w + w), :].copy()

            if self.rotation:
                (h, w, c) = img.shape
                angle = np.random.uniform(-self.rotation, self.rotation)
                RotMat = cv2.getRotationMatrix2D(center=(w, h), angle=angle, scale=1)
                img = cv2.warpAffine(img, RotMat, (w, h))
                label = cv2.warpAffine(label, RotMat, (w, h))

            if self.brightness_range:
                scale_bright = np.random.uniform(self.brightness_range[0], self.brightness_range[1])
                img = img * scale_bright

            label = label > 0.5
            X[i,] = img
            # Lưu class
            Y[i,] = label
        return X, Y



from tensorflow.keras.models import *


from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.utils import *

# print(os.listdir())


# def unet(pretrained_weights=None, input_size=( 256, 256, 3)):
#     inputs = Input(input_size)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
#     conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
#     pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
#     conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
#     pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
#     conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
#     pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
#     conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
#     drop4 = Dropout(0.5)(conv4)
#     pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
#
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
#     conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
#     drop5 = Dropout(0.5)(conv5)
#
#     up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(drop5))
#     merge6 = concatenate([drop4, up6], axis=3)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
#     conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)
#
#     up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv6))
#     merge7 = concatenate([conv3, up7], axis=3)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
#     conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)
#
#     up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv7))
#     merge8 = concatenate([conv2, up8], axis=3)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
#     conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)
#
#     up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
#         UpSampling2D(size=(2, 2))(conv8))
#     merge9 = concatenate([conv1, up9], axis=3)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
#     conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
#     conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)
#
#     model = Model(inputs=inputs, outputs=conv10)
#
#     model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
#
#     # model.summary()
#
#     if (pretrained_weights):
#         model.load_weights(pretrained_weights)
#
#     return model


import tensorflow as tf

INPUT_SHAPE = 256
OUTPUT_SHAPE = 256

def _downsample_cnn_block(block_input, channel, is_first = False):
  if is_first:
    conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(block_input)
    conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(conv1)
    return [block_input, conv1, conv2]
  else:
    maxpool = tf.keras.layers.MaxPool2D(pool_size=2)(block_input)
    conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(maxpool)
    conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(conv1)
    return [maxpool, conv1, conv2]

def _upsample_cnn_block(block_input, block_counterpart, channel, is_last = False):
  # Upsampling block
  uppool1 = tf.keras.layers.Convolution2DTranspose(channel, kernel_size=2, strides=2)(block_input)
  # Crop block counterpart
  shape_input = uppool1.shape[2]
  shape_counterpart = block_counterpart.shape[2]
  crop_size = int((shape_counterpart-shape_input)/2)
  # Có thể bỏ qua crop vì các nhánh đã bằng kích thước.
  block_counterpart_crop = tf.keras.layers.Cropping2D(cropping=((crop_size, crop_size), (crop_size, crop_size)))(block_counterpart)
  concat = tf.keras.layers.Concatenate(axis=-1)([block_counterpart_crop, uppool1])
  conv1 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(concat)
  conv2 = tf.keras.layers.Conv2D(filters=channel, kernel_size=3, strides=1, padding='same')(conv1)
  if is_last:
    conv3 = tf.keras.layers.Conv2D(filters=1, kernel_size=3, strides=1, padding='same', activation='sigmoid')(conv2)
    return [concat, conv1, conv2, conv3]
  return [uppool1, concat, conv1, conv2]
from tensorflow.keras.optimizers import Adam

def _create_model2():
  ds_block1 = _downsample_cnn_block(tf.keras.layers.Input(shape=(INPUT_SHAPE, INPUT_SHAPE, 3)), channel=64, is_first = True)
  ds_block2 = _downsample_cnn_block(ds_block1[-1], channel=128)
  ds_block3 = _downsample_cnn_block(ds_block2[-1], channel=256)
  ds_block4 = _downsample_cnn_block(ds_block3[-1], channel=512)
  ds_block5 = _downsample_cnn_block(ds_block4[-1], channel=1024)
  us_block4 = _upsample_cnn_block(ds_block5[-1], ds_block4[-1], channel=512)
  us_block3 = _upsample_cnn_block(us_block4[-1], ds_block3[-1], channel=256)
  us_block2 = _upsample_cnn_block(us_block3[-1], ds_block2[-1], channel=128)
  us_block1 = _upsample_cnn_block(us_block2[-1], ds_block1[-1], channel=64, is_last = True)
  model = tf.keras.models.Model(inputs = ds_block1[0], outputs = us_block1[-1])
  model.compile(optimizer=Adam(lr = 1e-4), loss='binary_crossentropy', metrics=['accuracy'])
  return model

model = _create_model2()


path_image = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/train/image/"
path_label = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/train/label/"

import os
image_paths = [path_image + file for file in os.listdir(path_image)]
label_paths = [path_label + file for file in os.listdir(path_image)]
train_img_paths, val_img_paths, train_label_paths, val_label_paths = train_test_split(image_paths, label_paths, test_size = 0.2)
train_generator = DataGenerator(
    all_filenames = list(zip(train_img_paths, train_label_paths)),
    labels = train_label_paths,
    batch_size = 8,
    input_dim = (256, 256),
    n_channels = 3,
    normalize = False,
    zoom_range = False,
    rotation = 90,
    brightness_range=[0.8, 1],
    shuffle = True
)

val_generator = DataGenerator(
    all_filenames = list(zip(val_img_paths, val_label_paths)),
    labels = val_label_paths,
    batch_size=8,
    input_dim=(256, 256),
    n_channels=3,
    normalize=False,
    zoom_range=False,
    rotation=90,
    brightness_range=[0.8, 1],
    shuffle=True
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.callbacks import ReduceLROnPlateau
import matplotlib.pyplot as plt
import cv2

model.summary()
earlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')
mcp_save = ModelCheckpoint("model_unet_v2.h5", save_best_only=True, monitor='val_loss', mode='min')
# reduce_lr_loss = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1, epsilon=1e-4, mode='min')

history = model.fit(train_generator,
          steps_per_epoch=len(train_generator),
          validation_data=val_generator,
          validation_steps=len(val_generator),
          epochs=50,
          callbacks=[earlyStopping, mcp_save]
          )

def display_training_curves(training, validation, title, subplot):
  ax = plt.subplot(subplot)
  ax.plot(training)
  ax.plot(validation)
  ax.set_title('model '+ title)
  ax.set_ylabel(title)
  ax.set_xlabel('epoch')
  ax.legend(['training', 'validation'])
plt.subplots(figsize=(10,10))
plt.tight_layout()
display_training_curves(history.history['accuracy'], history.history['val_accuracy'], 'accuracy', 211)
display_training_curves(history.history['loss'], history.history['val_loss'], 'loss', 212)