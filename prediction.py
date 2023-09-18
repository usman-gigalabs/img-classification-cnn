import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import numpy as np

# Load your trained model
model = load_model('model.keras')

# Load and preprocess a test image
img_path = 'animal-data/raw-img/gatto/1940.jpeg'
img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0  # Normalize pixel values to [0, 1]

# Make predictions
predictions = model.predict(img_array)

# Get the class label with the highest probability
predicted_class = np.argmax(predictions, axis=1)

# Class names and their transalations
class_mapping = {0: "cane", 1: "cavallo", 2: "elefante", 3: "farfalla", 4: "gallina", 5: "gatto", 6: "mucca", 7: "pecora", 8:"ragno", 9: "scoiattolo"}
translate = {"cane": "dog", "cavallo": "horse", "elefante": "elephant", "farfalla": "butterfly", "gallina": "chicken", "gatto": "cat", "mucca": "cow", "pecora": "sheep", "ragno":"spider", "scoiattolo": "squirrel"}

predicted_class_label = class_mapping.get(predicted_class[0], 'Unknown')
# Print the predicted class label
print('Predicted Class: {} => {}'.format(translate[predicted_class_label], predicted_class_label))
