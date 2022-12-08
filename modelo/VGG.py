import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense, BatchNormalization


train_datagen = ImageDataGenerator(rescale = 1./255,
                                       rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./MMAFEDB/train',
                                                  target_size = (48, 48),
                                                  batch_size = 128,
                                                  class_mode = 'categorical'
)


validation_set = train_datagen.flow_from_directory('./MMAFEDB/valid',
                                                  target_size = (48, 48),
                                                  batch_size = 128,
                                                  class_mode = 'categorical')

class_names = training_set.class_indices

num_classes = len(class_names)

model=Sequential([
    tf.keras.applications.VGG16(input_shape=(48,48,3),include_top=False,weights="imagenet"),
    Dropout(0.5),
    Flatten(),
    BatchNormalization(),
    Dense(32,kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(32,kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dropout(0.5),
    Dense(32,kernel_initializer='he_uniform'),
    BatchNormalization(),
    Activation('relu'),
    Dense(7,activation='softmax')
  ])

opt = Adam(lr=0.001)
model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])

model.summary()

checkpoint = ModelCheckpoint("vgg16.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)

hist = model.fit(generator=training_set, validation_data= validation_set, epochs=10,callbacks=[checkpoint], shuffle = True)
