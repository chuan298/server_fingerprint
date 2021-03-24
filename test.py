
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras.optimizers import *
from tensorflow.keras.callbacks import ModelCheckpoint, LearningRateScheduler
from tensorflow.keras import backend as keras
from tensorflow.keras.utils import *
def unet(pretrained_weights=None, input_size=( 256, 256, 3)):
    inputs = Input(input_size)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
    conv1 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool1)
    conv2 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool2)
    conv3 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool3)
    conv4 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)

    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(pool4)
    conv5 = Conv2D(1024, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)

    up6 = Conv2D(512, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(drop5))
    merge6 = concatenate([drop4, up6], axis=3)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge6)
    conv6 = Conv2D(512, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv6)

    up7 = Conv2D(256, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv6))
    merge7 = concatenate([conv3, up7], axis=3)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge7)
    conv7 = Conv2D(256, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv7)

    up8 = Conv2D(128, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv7))
    merge8 = concatenate([conv2, up8], axis=3)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge8)
    conv8 = Conv2D(128, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv8)

    up9 = Conv2D(64, 2, activation='relu', padding='same', kernel_initializer='he_normal')(
        UpSampling2D(size=(2, 2))(conv8))
    merge9 = concatenate([conv1, up9], axis=3)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(merge9)
    conv9 = Conv2D(64, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation='relu', padding='same', kernel_initializer='he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation='sigmoid')(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    model.compile(optimizer=Adam(lr=1e-4), loss='binary_crossentropy', metrics=['accuracy'])

    model.summary()

    if (pretrained_weights):
        model.load_weights(pretrained_weights)

    return model


def my_image_mask_generator(image_data_generator, mask_data_generator):
    train_generator = zip(image_data_generator, mask_data_generator)
    for (img, mask) in train_generator:
        yield (img, mask)

SEED_TRAIN = 100
SEED_VALID = 200

path_train_image = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/train/image/"
path_train_label = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/train/label/"
path_valid_image = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/valid/image/"
path_valid_label = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/valid/label/"

train_image_data_generator = ImageDataGenerator(
    rotation_range=90,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True,
).flow_from_directory(path_train_image, batch_size = 16, target_size = (256, 256), seed = SEED_TRAIN, class_mode=None)

train_mask_data_generator = ImageDataGenerator(
    rotation_range=90,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True
).flow_from_directory(path_train_label, batch_size = 16, target_size = (256, 256), seed = SEED_TRAIN, class_mode=None)

valid_image_data_generator = ImageDataGenerator(
    # width_shift_range = 0.1,
    # height_shift_range = 0.1,
    rotation_range = 90,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip= True,
    vertical_flip= True
).flow_from_directory(path_valid_image, batch_size = 16, target_size = (256, 256), seed = SEED_VALID, class_mode=None)

valid_mask_data_generator = ImageDataGenerator(
    rotation_range=90,
    featurewise_center=True,
    featurewise_std_normalization=True,
    horizontal_flip=True,
    vertical_flip=True
).flow_from_directory(path_valid_label, batch_size = 16, target_size = (256, 256), seed = SEED_VALID, class_mode=None)

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
# Your other related imports here...

# Create custom generator for training images and masks
train_generator = zip(train_image_data_generator, train_mask_data_generator)
# valid_generator = zip(valid_image_data_generator, valid_mask_data_generator)


model = unet()

# Compile your model here
model.compile(optimizer = Adam(lr = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])

earlyStopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, mode='min')
mcp_save = ModelCheckpoint("model_unet_v2.h5", save_best_only=True, monitor='val_loss', mode='min')
# Train your model here
history = model.fit(train_generator,
          steps_per_epoch=2000,
          # validation_data=valid_generator,
          # validation_steps=4153 // 16,
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










# import os
# path_image = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/image/"
# path_label = "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/label/"
#
#
# image_paths = []
# label_paths = []
#
# list_file = os.listdir("/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/train/image/")
# for i in range(len(list_file)):
#   # if i > 10: break
#   image_paths.append("/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/train/image/" + list_file[i])
#   label_paths.append("/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/train/label/" + list_file[i])
# print(len(image_paths), len(label_paths))


# import shutil
# import random
#
#
# dem = 0
#
# list_num = list(range(0, len(label_paths)))
# while(dem < 3988):
#   i = random.choice(list_num)
#   list_num.remove(i)
#   print(i)
#   try:
#     shutil.move(image_paths[i], "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/valid/image/")
#     shutil.move(label_paths[i], "/mnt/9dd20cb6-bc4b-413c-a3a2-81fb9435f460/data/data_do_an/data_unet/data_final_2/valid/label/")
#     dem += 1
#   except Exception as e:
#     print(e)