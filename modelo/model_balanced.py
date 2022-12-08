import os
import cv2
import numpy as np
import pandas as pd
from math import floor

import tensorflow as tf
import matplotlib.pyplot as plt


from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten


label_to_number = {
                    'angry': 0,
                    'disgust': 1,
                    'fear': 2,
                    'happy': 3,
                    'neutral': 4,
                    'sad': 5,
                    'surprise': 6
}


# Funcion para cargar las imágenes manualmente desde los directorios en un Dataframe
def loadDataFrameFromDirectory(input_path):
    
    expressions = os.listdir(input_path)
    n_exp = len(expressions)

    data = []
    labels = []

    for expression in expressions:

        for image in os.listdir(os.path.join(input_path, expression)):
            img = cv2.imread(os.path.join(input_path, expression, image))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            data.append(img)
            labels.append(label_to_number[expression])

    df = pd.DataFrame({'image': data, 'label': labels})
    df['image'] = df.image.apply(lambda x: x.reshape(48, 48, 1).astype('float32'))

    return df


# Función para modificar las imágenes de las que se hace oversampling
def dataAugmentation(img):

    image = tf.image.random_flip_left_right(img.reshape(48, 48, 1))

    image = tf.image.resize_with_crop_or_pad(image, 48 + 12, 48 + 12)
    image = tf.image.random_crop(image, size=[48, 48, 1])

    image = tf.image.random_brightness(image, max_delta=0.5)

    return image.numpy()


# Funcion para ajustar todos los conjuntos a la media de ellos
def balanceSample(df):

    counts = df.label.value_counts()
    mean = floor(counts.mean())

    diff = mean - counts

    print('####### STARTING DATA AUGMENTATION #######')

    for emotion, difference in diff.items():

        if difference > 0:
            print(f'AUGMENTING DATA FOR: {emotion}')
            #Oversamplig.
            sample = df.query('label==@emotion').sample(difference, replace = True)
            sample['image'] = sample.image.apply(dataAugmentation)
            #Añadimos las duplicadas aumentadas. 
            df = pd.concat([df, sample])

        else:
            print(f'REDUCING DATA FOR: {emotion}')
            #Undersampling. 
            sample = df.query('label==@emotion').sample(abs(difference), replace = False)
            # sample['image'] = sample.image.apply(lambda x : data_aug(x.reshape(48,48,1)))
            #Borramos las que sobran
            df = df.drop(sample.index)

    return df


# Main
if __name__ == '__main__':


    train_set = loadDataFrameFromDirectory('MMAFEDB/train')
    validation_set = loadDataFrameFromDirectory('MMAFEDB/valid')
    test_set = loadDataFrameFromDirectory('MMAFEDB/test')


    train_balanced = balanceSample(train_set)

    print(train_set.label.value_counts())
    print(train_balanced)

    num_classes = len(train_set.label.unique())

    model = Sequential([ 

      BatchNormalization(),
      
      Conv2D(64, 3, padding = 'same', activation = 'relu'),
      BatchNormalization(),
      Conv2D(64, 3, padding = 'same', activation = 'relu'),
      BatchNormalization(),
      MaxPooling2D(pool_size = (2, 2)),
      Dropout(0.4),

      Conv2D(128, 3, padding = 'same', activation = 'relu'),
      BatchNormalization(),
      Conv2D(128, 3, padding = 'same', activation = 'relu'),
      BatchNormalization(),
      MaxPooling2D(pool_size = (2, 2)),
      Dropout(0.4),

      Conv2D(256, 3, padding = 'same', activation = 'relu'),
      BatchNormalization(),
      Conv2D(256, 3, padding = 'same', activation = 'relu'),
      BatchNormalization(),
      MaxPooling2D(pool_size = (2, 2)),
      Dropout(0.4),

      Flatten(),
      
      Dense(128, activation = 'relu'),
      BatchNormalization(),
      Dropout(0.4),

      
      Dense(num_classes, activation = 'softmax')
    ])

    model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001),
              loss = 'sparse_categorical_crossentropy',
              metrics = ['accuracy'])

    img_array = train_balanced.image.apply(lambda x: x.reshape(48, 48, 1).astype('float32'))
    img_array = np.stack(img_array, axis=0)

    val_array = validation_set.image.apply(lambda x: x.reshape(48, 48, 1).astype('float32'))
    val_array = np.stack(val_array, axis=0)

    train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True,
    )
    
    train_datagen.fit(img_array)

    checkpoint = ModelCheckpoint("model.h5", monitor = 'val_loss', batch_size = 128, verbose = 1, save_best_only = True, save_weights_only = False, mode = 'auto')

    lr_scheduler = ReduceLROnPlateau(
    monitor='val_accuracy',
    factor=0.5,
    patience=7,
    min_lr=1e-7,
    verbose=1,
)

    history = model.fit(
      train_datagen.flow(img_array, train_balanced.label, batch_size = 128),
      #validation_split = 0.2,
      validation_data = (val_array, validation_set.label),
      epochs = 30,
      shuffle = True,
      callbacks = [lr_scheduler]
    )

    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']

    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs_range = range(30)

    plt.figure(figsize=(8, 8))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Training Accuracy')
    plt.plot(epochs_range, val_acc, label='Validation Accuracy')
    plt.legend(loc='lower right')
    plt.title('Training and Validation Accuracy')

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.legend(loc='upper right')
    plt.title('Training and Validation Loss')
    plt.show()
    