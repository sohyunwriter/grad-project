# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))

# Any results you write to the current directory are saved as output.
from PIL import Image
from keras.preprocessing import image
import os
import numpy as np

# resize image (224, 224)
img_rows, img_cols = 128, 128

#listing = os.listdir('../input/diabetic-retinopathy-resized/resized_train/resized_train')

data = pd.read_csv("../input/diabetic-retinopathy-resized/trainLabels_cropped.csv")
data['path'] = [ "../input/diabetic-retinopathy-resized/resized_train_cropped/resized_train_cropped/" + i+".jpeg" for i in data['image'].values]
data.head()

input_shape = (128, 128, 3)

import keras
from keras.models import load_model
from keras import optimizers
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical

import matplotlib.pyplot as plt
%matplotlib inline

from tqdm import tqdm

# Transforming high resolution images to low resolution to meet up memory constraints
train_image = []
y_train = []
#for i in tqdm(range(data.shape[0])):

for i in tqdm(range(data.shape[0])):
    img = image.load_img(data['path'][i],target_size=input_shape,interpolation='nearest')
    img = image.img_to_array(img)
    img = img.astype('float32')
    img = img/255
    train_image.append(img)
    y_train.append(data['level'][i])
x_train = np.array(train_image)

x_train

#y_train = data['level']
y_train = keras.utils.to_categorical(y_train, 5)

del data, train_image
import gc;
gc.collect()

x_train.shape[0]

x_train.shape

row, col, pixel = x_train.shape[1:]

row, col, pixel

import keras
from keras.models import Model
from keras.layers import Input, Dense, TimeDistributed
from keras.layers import LSTM
x = Input(shape=(row, col, pixel))

row_hidden = 128
col_hidden = 128
encoded_rows = TimeDistributed(LSTM(row_hidden))(x)

encoded_columns = LSTM(col_hidden)(encoded_rows)

num_classes = 5
batch_size = 32
epochs = 3
prediction = Dense(num_classes, activation='softmax')(encoded_columns)
model = Model(x, prediction)
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(x_train, y_train, test_size = 0.15, random_state=42)

model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))

scores = model.evaluate(X_test, Y_test, verbose=0)
print('Test loss: ', scores[0])
print('Test accuracy: ', scores[1])

from keras.models import load_model
model.save('my_rnn_model.h5')

