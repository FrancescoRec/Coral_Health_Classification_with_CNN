"""
Hello President of IslandCity!
Here I have implemented the function that classifies the images in the dataset
and stores the results in the database.
You can run the script with the path of the folder that contains the images you want to test as an argument.
The model should be in the same folder of this script.

Usage:
python corals_classifier.py path_to_images (in this case just run python corals_classifier.py test_images)

Let me know if you need any help. Be green!
"""

import sqlite3
import numpy as np
from tensorflow.keras.models import load_model
import cv2
from imutils import paths
import sys

def create_connection():
    conn = sqlite3.connect('results.db')
    cursor = conn.cursor()
    cursor.execute('''CREATE TABLE IF NOT EXISTS sample_results
                    (id TEXT, class TEXT)''')
    return conn, cursor

def read_and_preprocess_image(image_path):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  
    return image

def classify_and_store(file_path):
    model = load_model('model_corals.keras')
    
    conn, cursor = create_connection()
    imagePaths = list(paths.list_images(file_path))

    for imagePath in imagePaths:
        sample_id = imagePath
        sample_image = read_and_preprocess_image(imagePath)
        sample_image = np.expand_dims(sample_image, axis=0)

        y_pred = model.predict(sample_image)

        if y_pred >0.5:
            predicted_label = 'healthy'
        else:
            predicted_label = 'bleached'

        cursor.execute('''INSERT INTO sample_results (id, class) VALUES (?, ?)''', (sample_id, predicted_label))
    
    conn.commit()
    conn.close()

if __name__ == '__main__':
    if len(sys.argv) != 2:
        sys.exit(1)
    
    classify_and_store(sys.argv[1])
