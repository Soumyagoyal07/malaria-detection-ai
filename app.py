from flask import Flask, render_template, request, jsonify
import numpy as np
import os
from datetime import datetime
from PIL import Image
import tensorflow as tf

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

model = None
model_error = None
model_path = None

class_names = ["Parasitized","Uninfected"]

try:
    for file in os.listdir():
        if file.endswith(".h5"):
            model_path = os.path.join(os.getcwd(),file)
            break

    if model_path is None:
        model_path = "best_malaria_model.h5"

    model = tf.keras.models.load_model(model_path)

    model.compile(
        optimizer="adam",
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    print("Model loaded successfully")
    print("Model path:",model_path)
    print("Input shape:",model.input_shape)
    print("Output shape:",model.output_shape)

except Exception as e:
    model_error = str(e)
    print("Model loading failed:",model_error)

def prepare_image(image):

    img = Image.open(image)

    if img.mode != "RGB":
        img = img.convert("RGB")

    input_shape = model.input_shape

    if len(input_shape) == 4:
        target_size = (input_shape[1],input_shape[2])
    else:
        target_size = (128,128)

    img = img.resize(target_size)

    img = np.array(img)

    img = img/255.0

    img = np.expand_dims(img,axis=0)

    return img

@app.route("/predict",methods=["POST"])
def predict():

    if model is None:
        return jsonify({
            "success":False,
            "error":"Model not loaded"
        }),500

    if "file" not in request.files:
        return jsonify({
            "success":False,
            "error":"No file uploaded"
        }),400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({
            "success":False,
            "error":"Empty filename"
        }),400

    try:

        processed_image = prepare_image(file)

        prediction = model.predict(processed_image,verbose=0)

        prob = float(prediction[0][0])

        if prob > 0.5:
            label = class_names[0]
            confidence = prob
        else:
            label = class_names[1]
            confidence = 1 - prob

        response = {
            "success":True,
            "prediction":label,
            "confidence":round(confidence*100,2),
            "probabilities":{
                "Parasitized":round(prob*100,2),
                "Uninfected":round((1-prob)*100,2)
            },
            "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        return jsonify(response)

    except Exception as e:
        return jsonify({
            "success":False,
            "error":str(e)
        }),500

@app.route("/")
def home():

    if os.path.exists("templates/index.html"):
        return render_template("index.html")

    status = "Model Loaded" if model else "Model Not Loaded"

    return f"""
    <html>
    <head>
    <title>Malaria Detection</title>
    <style>
    body {{
        font-family: Arial;
        text-align: center;
        background: linear-gradient(135deg,#667eea,#764ba2);
        color:white;
        padding:50px;
    }}
    </style>
    </head>
    <body>
    <h1>Malaria Detection System</h1>
    <p>Server running on port 5000</p>
    <p>Status: {status}</p>
    </body>
    </html>
    """

@app.route("/health")
def health():

    return jsonify({
        "server":"running",
        "model_loaded": model is not None
    })

if __name__ == "__main__":

    os.makedirs("templates",exist_ok=True)
    os.makedirs("static",exist_ok=True)

    if model:
        print("Model loaded successfully")
    else:
        print("Model not loaded")

    print("Server running at http://127.0.0.1:5000")

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True,
        use_reloader=False
    )