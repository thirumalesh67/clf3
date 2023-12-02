from flask import Flask, render_template, request
import os
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.resnet50 import preprocess_input
import numpy as np
import requests
import shutil

app = Flask(__name__)

# Load the trained model
model = None

def load_trained_model():
    global model
    if os.path.exists("trained_model.h5"):
        print("file exists")
    else:
        r = requests.get("https://clfsatimg.blob.core.windows.net/model/trained_model.h5", stream=True)
        with open("trained_model.h5", "wb") as f:
            shutil.copyfileobj(r.raw, f)

    model_path = 'trained_model.h5'  # Update with the actual path to your saved model
    model = load_model(model_path)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img = image.img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Load the model if it is not already loaded
        if model is None:
            load_trained_model()

        # Get the image file from the request
        img_file = request.files['image']
        img_path = 'temp_image.jpg'

        # Save the image file temporarily
        img_file.save(img_path)

        # Preprocess the image
        img = preprocess_image(img_path)

        # Make prediction
        predictions = model.predict(img)
        predicted_class = np.argmax(predictions)

        # Get class names from your dataset
        class_names = [
            'agricultural', 'airplane', 'baseballdiamond', 'beach', 'buildings',
            'chaparral', 'denseresidential', 'forest', 'freeway', 'golfcourse',
            'harbor', 'intersection', 'mediumresidential', 'mobilehomepark',
            'overpass', 'parkinglot', 'river', 'runway', 'sparseresidential',
            'storagetanks', 'tenniscourts'
        ]
        # Update with your actual class names

        result = {'class': class_names[predicted_class], 'confidence': float(predictions[0][predicted_class])}

        # Remove the temporary image file
        os.remove(img_path)

        return render_template('result.html', result=result)

    except Exception as e:
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    # Get the port number from the environment variable or use a default value
    port = int(os.environ.get('PORT', 5001))

    # Load the trained model
    load_trained_model()

    # Run the Flask application
    app.run(host='0.0.0.0', port=port, debug=True)
