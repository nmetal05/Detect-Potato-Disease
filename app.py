from flask import Flask, request,render_template,send_from_directory
from werkzeug.utils import secure_filename
import numpy as np
import tensorflow as tf
import os
from PIL import Image

os.environ["CUDA_VISIBLE_DEVICES"]="-1"

app = Flask(__name__)

class_names = ['Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy']

def predict(model, img):
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)

    predictions_arr = [round(100*i,2) for i in predictions[0]]
    predicted_class = class_names[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return predicted_class, predictions_arr

model = tf.keras.models.load_model('potato_model.h5',compile=False)

@app.route('/predict',methods=['POST'])
def predict_image():
    if request.method == 'POST':
        print("request received")
        print(request.files)
        file = request.files['image']
        filename = secure_filename(file.filename)
        img = Image.open(file.stream)
        img = img.resize((256,256))
        img_array = np.array(img)
        predicted_class,predictions = predict(model,img_array)
        response = {"predicted_class": f"{predicted_class}" ,"early": f"{predictions[0]:.2f}%","late": f"{predictions[1]:.2f}%","healthy": f"{predictions[2]:.2f}%"}
        img.save(os.path.join('uploads',filename))
        print(response)
        print(f"Predictions : {predictions}")
        return render_template('result.html',result=response,image_path=f'uploads/{filename}')

@app.route('/uploads/<path:filename>')
def serve_uploaded_file(filename):
    return send_from_directory('uploads', filename)

@app.route('/')
def index():
    return render_template('index.html',mimetype='text/html')

if __name__ == '__main__':
    app.run(debug=True)
