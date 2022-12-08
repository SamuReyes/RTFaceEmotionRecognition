import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Se introduce a partir del directorio de entrada el conjunto de test
test_datagen = ImageDataGenerator()
test_set = test_datagen.flow_from_directory('./MMAFEDB/test', target_size = (48, 48), batch_size = 128, class_mode = 'categorical', color_mode = 'rgb')

test_set = tf.keras.utils.image_dataset_from_directory(
    './MMAFEDB/test',
    color_mode = 'grayscale',
    shuffle = True,
    batch_size = 128,
    image_size = (48, 48),
    seed = 123
)

class_names = test_set.class_names

final_model = tf.keras.models.load_model('model_rgb_oversampling.h5')

# Arquitectura de la red
print(final_model.summary())

# Evaluacion en conjunto de test
test_results = final_model.evaluate(test_set, batch_size = 128, verbose = 0)
print(f'Test results - Loss: {test_results[0]} - Accuracy: {test_results[1]}')
