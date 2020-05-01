# coding=utf-8
import numpy as np

import tensorflow as tf
# keras
from keras.applications.imagenet_utils import preprocess_input, decode_predictions
from keras.models import load_model
from keras.preprocessing import image
from keras import models
#from tensorflow.keras.models import load_model
from keras.applications.resnet50 import decode_predictions

# flask utils
from flask import Flask, render_template, request, url_for, redirect
from werkzeug.utils import secure_filename
from gevent.pywsgi import WSGIServer
from keras import backend as K

#UPLOAD_FOLDER = '/path/to/the/uploads'
#ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif'])

# define flask app
app = Flask(__name__)
app.debug = False
#app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# model
MODEL_PATH = 'models/my_rnn_model.h5'

from keras.applications.resnet50 import ResNet50
#model = ResNet50(weights='imagenet')
#model = load_model(MODEL_PATH)
#model._make_predict_function()

model = None

model = models.load_model(MODEL_PATH)
model.summary()
print('loaded the model')
model._make_predict_function()

#모델 추가
print('Model loaded. Check http://127.0.0.1:5000/')

CATEGORIES = ['0(NO DR)', '1(Mild)', '2(Moderate)', '3(Severe)', '4(Proliferative DR)']

# predict
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(128,128))
    # preprocessing image
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x, mode='caffe')
    preds = model.predict(x)
    print(preds)
    print(CATEGORIES[np.argmax(preds)])
    return CATEGORIES[np.argmax(preds)]

# upload
@app.route('/predict', methods=['GET', 'POST'])
def upload():
    if request.method=='POST':
        f = request.files['file']
        file_path = secure_filename('img/' + f.filename)
        print(file_path)
        f.save(file_path)
        result = model_predict(file_path, model)
        return result
    return None

@app.route('/')
def index():
    return render_template('first.html')

if __name__ == '__main__':
    app.run(threaded=False)
    #http_server = WSGIServer(('', 5000), app)
    #http_server.serve_forever()