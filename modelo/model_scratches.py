import matplotlib.pyplot as plt
from tensorflow.keras import layers
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, BatchNormalization, MaxPooling2D, Dropout, Dense, Flatten


# Carga de datasets
train_datagen = ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.15,
    height_shift_range=0.15,
    shear_range=0.15,
    zoom_range=0.15,
    horizontal_flip=True)

validation_datagen = ImageDataGenerator()

training_set = train_datagen.flow_from_directory('./MMAFEDB/train',
                                                  target_size = (48, 48),
                                                  batch_size = 128,
                                                  class_mode = 'categorical',
                                                  color_mode = 'rgb'
)


validation_set = train_datagen.flow_from_directory('./MMAFEDB/valid',
                                                  target_size = (48, 48),
                                                  batch_size = 128,
                                                  class_mode = 'categorical',
                                                  color_mode = 'rgb')

class_names = training_set.class_indices

print(class_names)
print(type(class_names))


num_classes = len(class_names)

'''
model = Sequential([ 

  Conv2D(16, 3, padding = 'same', activation = 'relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size = (2, 2)),
  Dropout(0.2),
  
  Conv2D(32, 3, padding = 'same', activation = 'relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size = (2, 2)),
  Dropout(0.2),

  Conv2D(64, 3, padding = 'same', activation = 'relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size = (2, 2)),
  Dropout(0.2),

  Conv2D(128, 3, padding = 'same', activation = 'relu'),
  BatchNormalization(),
  MaxPooling2D(pool_size = (2, 2)),
  Dropout(0.2),

  Flatten(),
  
  Dense(128, activation = 'relu'),
  Dropout(0.2),
  Dense(num_classes, activation = 'softmax')
])'''

model = Sequential([ 
  
  BatchNormalization(input_shape = (48, 48, 3)),

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

model.compile(optimizer = 'adam',
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])


epochs=30

checkpoint = ModelCheckpoint("model.h5", monitor='val_accuracy', verbose=1, save_best_only=True, save_weights_only=False, mode='auto')

history = model.fit(
  training_set,
  validation_data = validation_set,
  epochs = epochs,
  shuffle = True,
  callbacks = [checkpoint]
)

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)

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
