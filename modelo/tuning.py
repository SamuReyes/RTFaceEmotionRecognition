import keras_tuner as kt
from tensorflow import keras
from keras.callbacks import ModelCheckpoint
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dropout, Dense, BatchNormalization


# Se cargan de los directorios los datasets
train_datagen = ImageDataGenerator(rescale = 1./255,
                                    rotation_range=15,
                                    width_shift_range=0.15,
                                    height_shift_range=0.15,
                                    shear_range=0.15,
                                    zoom_range=0.15,
                                    horizontal_flip=True)

validation_datagen = ImageDataGenerator(rescale = 1./255)
test_set = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('./MMAFEDB/train',
                                                  target_size = (48, 48),
                                                  batch_size = 64,
                                                  class_mode = 'categorical',
                                                  color_mode = 'grayscale'
  )

validation_set = validation_datagen.flow_from_directory('./MMAFEDB/valid',
                                                  target_size = (48, 48),
                                                  batch_size = 64,
                                                  class_mode = 'categorical',
                                                  color_mode = 'grayscale')

test_set = validation_datagen.flow_from_directory('./MMAFEDB/test',
                                                  target_size = (48, 48),
                                                  batch_size = 64,
                                                  class_mode = 'categorical',
                                                  color_mode = 'grayscale')

# Número de clases
class_names = training_set.class_indices
num_classes = len(class_names)


# Build model de la arquitectura 1
def build_model_1(hp):
    
    model = keras.models.Sequential()

    # Capa de entrada
    model.add(Conv2D(hp.Int('input_units', min_value=32, max_value=256, step=32), (3, 3), input_shape=(48, 48, 1), activation = 'relu', padding = 'same'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides = hp.Choice('strides_init', values = [1, 2])))

    # Capas convolucionales de 0 a 4
    for i in range(hp.Int('n_layers', 0, 4)):
        # Capa convloucional
        model.add(Conv2D(hp.Int(f'conv_{i}_units', min_value=32, max_value=256, step=32), (3, 3), activation = 'relu', padding = 'same'))

        #Con una o ninguna capa de maxpooling
        for j in range(hp.Choice(f'pooling_layers{i}', values = [0, 1])):
            model.add(MaxPooling2D(pool_size=(2, 2), strides = hp.Choice(f'strides_{i}', values = [1, 2])))

        for k in range(hp.Choice(f'dropout_layers_{i}', values = [0, 1])):
            model.add(Dropout(0.5, name = f'dropout_{i}'))

    # Capa para convertir la salida en un vector de entrada a las capas densas
    model.add(Flatten())

    # Se incluyed de una a cuatro capas densas
    for i in range(hp.Int('n_connections', 1, 2)):
        
        model.add(Dense(hp.Choice(name = f'{i}_nodes', values=[128, 256, 512, 1024]), activation = 'relu'))

    model.add(Dense(num_classes, activation = 'softmax'))

    model.compile(optimizer="adam", loss = 'categorical_crossentropy', metrics=["accuracy"])

    return model


# Build model de la arquitectuira 2
def build_model_2(hp):
    
    model = Sequential()

    filters = 64

    model.add(Conv2D(name = f'conv_layer_entry', filters = filters, kernel_size = 3, padding = 'same', activation = 'relu', input_shape=(48, 48, 1)))
    model.add(BatchNormalization(name = 'batch_normalization_entry_a'))
    model.add(Conv2D(name = f'conv_layer_entry_b', filters = filters, kernel_size = 3, padding = 'same', activation = 'relu'))
    model.add(BatchNormalization(name = f'batch_normalization_entry_b'))
    model.add(MaxPooling2D(pool_size = 2, strides = 2, name = f'pool_entry'))
    model.add(Dropout(name = f'drop_entry', rate = 0.4))

    for i in range(hp.Int('n_layers', 0, 2)):
        
        filters *= 2

        model.add(Conv2D(name = f'conv_layer_a{i}', filters = filters, kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(BatchNormalization(name = f'batch_normalization_a{i}'))
        model.add(Conv2D(name = f'conv_layer_b{i}', filters = filters, kernel_size = 3, padding = 'same', activation = 'relu'))
        model.add(BatchNormalization(name = f'batch_normalization_b{i}'))
        model.add(MaxPooling2D(pool_size = 2, strides = 2, name = f'pool_{i}'))
        model.add(Dropout(name = f'drop_{i}', rate = 0.4))


    model.add(Flatten(name = 'flatten'))

    model.add(Dense(units = 128, name = 'dense_1', activation = 'relu'))
    model.add(Dropout(name = f'drop_dense_1', rate = 0.5))

    model.add(BatchNormalization(name = 'batch_normalization_dense'))

    model.add(Dense(units = len(class_names), name = 'exit', activation = 'softmax'))

    model.compile(optimizer="adam", loss = 'categorical_crossentropy', metrics=["accuracy"])

    return model


if __name__ == '__main__':

    # Se inicializa el tuner con los valores especificados en build_model()
    tuner = kt.Hyperband(
        hypermodel = build_model_2,
        max_epochs = 5,
        hyperband_iterations = 3,
        objective = 'val_accuracy',
        factor = 3,
        seed = 123,
        overwrite = True,
        directory = './logs/')

    # Se buscan las mejores configuraciones de hiperparametros
    tuner.search(training_set, validation_data = validation_set, verbose = 1)

    # Se define el criterio de parada
    model_checkpoint = ModelCheckpoint(
        filepath = './checkpoint/',
        save_weights_only = True,
        monitor = 'accuracy',
        save_best_only = True)

    models = tuner.get_best_models(num_models=3)

    i = 0

    # Se reentrenan los mejores modelos 20 epocas
    for model in models:
        model.fit(training_set, epochs = 20, verbose = 1,  validation_data = validation_set,  shuffle = True, validation_freq = 1,  callbacks = [model_checkpoint])

        print (model.summary())

        # Evaluate the best model.
        loss, accuracy = model.evaluate(test_set)
        print('loss:', loss)
        print('accuracy:', accuracy)

        model.save(f'./model{i}.h5')
        i+=1
