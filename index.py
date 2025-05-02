from flask import Flask, request, jsonify
from keras.models import load_model
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = './model/model.h5'  
model = load_model(MODEL_PATH)

@app.route('/', methods=['POST'])
def predict():
    
    imagefile = request.files['imagefile']

    image_path = "./images/" + imagefile.filename
    imagefile.save(image_path)

    # Preprocessing
    image = load_img(image_path, target_size=(150, 150))  
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = image / 255.0  

    # Prediction
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions[0])  
    probability = float(np.max(predictions[0])) * 100  

    class_labels = ['cat', 'dog', 'wild'] 
    label = class_labels[predicted_class]

    result = {
        'label': label,
        'probability': probability
    }

    return jsonify(result)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080)
