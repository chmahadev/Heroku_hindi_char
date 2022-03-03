
from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow import keras
import cv2
import base64

# Initialize the app
app = Flask(__name__)

# Load prebuilt model
model = keras.models.load_model('base_model_93.4per.h5')
label = {'character_10_yna': 0,'character_11_taamatar': 1,
 'character_12_thaa': 2,
 'character_13_daa': 3,
 'character_14_dhaa': 4,
 'character_15_adna': 5,
 'character_16_tabala': 6,
 'character_17_tha': 7,
 'character_18_da': 8,
 'character_19_dha': 9,
 'character_1_ka': 10,
 'character_20_na': 11,
 'character_21_pa': 12,
 'character_22_pha': 13,
 'character_23_ba': 14,
 'character_24_bha': 15,
 'character_25_ma': 16,
 'character_26_yaw': 17,
 'character_27_ra': 18,
 'character_28_la': 19,
 'character_29_waw': 20,
 'character_2_kha': 21,
 'character_30_motosaw': 22,
 'character_31_petchiryakha': 23,
 'character_32_patalosaw': 24,
 'character_33_ha': 25,
 'character_34_chhya': 26,
 'character_35_tra': 27,
 'character_36_gya': 28,
 'character_3_ga': 29,
 'character_4_gha': 30,
 'character_5_kna': 31,
 'character_6_cha': 32,
 'character_7_chha': 33,
 'character_8_ja': 34,
 'character_9_jha': 35,
 'digit_0': 36,
 'digit_1': 37,
 'digit_2': 38,
 'digit_3': 39,
 'digit_4': 40,
 'digit_5': 41,
 'digit_6': 42,
 'digit_7': 43,
 'digit_8': 44,
 'digit_9': 45}

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
