
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64

# Initialize the app
app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model('4_model.h5')
label = {0: 'yna',1:'taamatar',
 2: 'thaa',
 3:'daa',
 4: 'dhaa',
 5: 'adna',
 6: 'tabala',
 7: 'tha',
 8: 'da',
 9: 'dha',
 10:'ka',
 11: 'na',
 12: 'pa',
 13: 'pha',
 14: 'ba',
 15: 'bha',
 16: 'ma',
 17: 'yaw',
 18: 'ra',
 19: 'la',
 20: 'waw',
 21: 'kha',
 22: 'motosaw',
 23: 'petchiryakha',
 24: 'patalosaw',
 25: 'ha',
 26: 'chhya',
 27: 'tra',
 28: 'gya',
 29: 'ga',
 30: 'gha',
 31: 'kna',
 32: 'cha',
 33: 'chha',
 34: 'ja',
 35: 'jha',
 36: '0',
 37: '1',
 38: '2',
 39: '3',
 40: '4',
 41: '5',
 42: '6',
 43: '7',
 44: '8',
 45: '9'}

# Handle GET request
@app.route('/', methods=['GET'])
def drawing():
    return render_template('index.html')

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
        prediction = label[np.argmax(model.predict(img))]
        print(f"Prediction Result : {str(prediction)}")
        return render_template('index.html', response=str(prediction), canvasdata=canvasdata, success=True)
    except Exception as e:
        return render_template('index.html', response=str(e), canvasdata=canvasdata)
