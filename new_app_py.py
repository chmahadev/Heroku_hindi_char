
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64

# Initialize the app
app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model('base_model_93.4per.h5')

# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('index.html5')

# Handle POST request
@app.route('/', methods=['POST'])
def canvas():
    # Recieve base64 data from the user form
    canvasdata = request.form['canvasimg']
    encoded_data = request.form['canvasimg'].split(',')[1]

    # Decode base64 image to python array
    nparr = np.fromstring(base64.b64decode(encoded_data), np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Convert 3 channel image (RGB) to 1 channel image (GRAY)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Resizing the imagec into (32, 32) shape
    gray_image = cv2.resize(gray_image, (32, 32), interpolation=cv2.INTER_LINEAR)

    # Expand to numpy array dimenstion to (1, 32, 32)
    img = np.expand_dims(gray_image, axis=0)

    try:
        prediction = np.argmax(model.predict(img))
        print(f"Prediction Result : {str(prediction)}")
        return render_template('index.html5', response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template('index.html5', response=str(e), canvasdata=canvasdata)
