import numpy as np
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

model = tf.keras.models.load_model('saved_model/car_model')

class_names = ['beige', 'black', 'blue', 
               'brown', 'gold', 'green', 
               'grey', 'orange', 'pink', 
               'purple', 'red', 'silver', 
               'tan', 'white', 'yellow'
              ]

# perform prediction on a single image.
def predict_image(tfmodel, file_path) -> dict:
    img_height = 180
    img_width = 180
    
    img = tf.keras.utils.load_img(
        file_path, target_size=(img_height, img_width)
    )
    
    img_array = tf.keras.utils.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0)
    
    predictions = tfmodel.predict(img_array)
    score = tf.nn.softmax(predictions[0])

    predicted_class = {
        'colour' : class_names[np.argmax(score)],
        'conf' : 100 * np.max(score)
    }

    return predicted_class