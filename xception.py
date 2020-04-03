import os
import argparse
import time
import random
import keras
import nsml
import numpy as np
import tensorflow as tf

from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from nsml.constants import DATASET_PATH, GPU_NUM
from keras.utils import np_utils
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

from dataprocessing import image_preprocessing, dataset_loader
from keras.applications.xception import Xception

## setting values of preprocessing parameters
RESIZE = 10.
RESCALE = True

TARGET_RESOLUTION = int(3900 // RESIZE), int(3072 // RESIZE)


# learning_rate = 1e-3

def bind_model(model, model1):
    def save(dir_name):
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model'))
        os.makedirs(dir_name, exist_ok=True)
        model.save_weights(os.path.join(dir_name, 'model1'))
        # model.save_weights(file_path,'model')
        print('model saved!')
        # model1.save_weights(os.path.join(dir_name, 'model1'))

    def load(dir_name):
        try:
            model.load_weights(os.path.join(dir_name, 'model'))
            print('model loaded!')
        except:
            model1.load_weights(os.path.join(dir_name, 'model1'))
            print('model1 loaded!')

    def infer(data,
              target_resolution=TARGET_RESOLUTION,
              normalize=RESCALE):  ## test mode
        ##### DO NOT CHANGE ORDER OF TEST DATA #####
        X = []
        for i, d in enumerate(data):
            # test 데이터를 training 데이터와 같이 전처리 하기
            X.append(image_preprocessing(d,
                                         target_resolution,
                                         normalize))

        X = np.array(X)

        pred = model.predict(X)
        pred = np.argmax(pred, axis=1)  # 모델 예측 결과: 0-3

        pred1 = model1.predict(X)
        pred1 = np.argmax(pred1, axis=1)
        pred = 0.7 * pred + 0.3 * pred1
        pred = np.argmax(pred, axis=1)
        print('Prediction done!\n Saving the result...')
        return pred

    nsml.bind(save=save, load=load, infer=infer)


if __name__ == '__main__':
    args = argparse.ArgumentParser()

    # hyperparameters
    args.add_argument('--epoch', type=int, default=2)  # epoch 수 설정
    args.add_argument('--batch_size', type=int, default=16)  # batch size 설정
    args.add_argument('--num_classes', type=int, default=4)  # DO NOT CHANGE num_classes, class 수는 항상 4

    # DONOTCHANGE: They are reserved for nsml
    args.add_argument('--mode', type=str, default='train', help='submit일때 해당값이 test로 설정됩니다.')
    args.add_argument('--iteration', type=str, default='0',
                      help='fork 명령어를 입력할때의 체크포인트로 설정됩니다. 체크포인트 옵션을 안주면 마지막 wall time 의 model 을 가져옵니다.')
    args.add_argument('--pause', type=int, default=0, help='model 을 load 할때 1로 설정됩니다.')

    config = args.parse_args()

    seed = 1234
    np.random.seed(seed)

    # training parameters
    nb_epoch = config.epoch
    batch_size = config.batch_size
    num_classes = config.num_classes

    """ Model """
    # make a model
    h, w = int(3072 // RESIZE), int(3900 // RESIZE)
    model = Xception(input_shape=(307, 390, 3), classes=4, include_top=True, weights=None, input_tensor=None,
                     pooling=None)
    model1 = Xception(input_shape=(307, 390, 3), classes=4, include_top=True, weights=None, input_tensor=None,
                      pooling=None)
    # model = DenseNet169(include_top= True, weights = None, input_shape = (300,300,3), classes = 4)
    # model = EfficientNetB4(include_top= True, weights = None, input_shape = (307,390,3), classes = 4)

    optimizer = optimizers.Adam(lr=1e-4)

    model.compile(loss='categorical_crossentropy',
                  optimizer=optimizer,
                  metrics=['categorical_accuracy'])

    model.summary()

    model1.compile(loss='categorical_crossentropy',
                   optimizer=optimizer,
                   metrics=['categorical_accuracy'])

    model1.summary()

    bind_model(model, model1)

    # binding a model
    # bind_model(model)

    if config.pause:  ## test mode일 때
        print('Inferring Start...')
        nsml.paused(scope=locals())

    if config.mode == 'train':  ### training mode일 때
        print('Training Start...')

        img_path = DATASET_PATH + '/train/'

        # for 10
        images, labels = dataset_loader(img_path,
                                        TARGET_RESOLUTION,
                                        normalize=RESCALE)

        nsml.load(checkpoint='best', session='team120/KHD2019_FUNDUS/135')
        nsml.save('best')

        # data 섞기
        total_size = images.shape[0]
        perm = np.random.permutation(total_size)
        x = images[perm]
        y = labels[perm]

        # train, validation split
        train_val_ratio = 0.8  # 0.8
        tmp = int(total_size * train_val_ratio)

        x_train = x[:tmp]
        x_val = x[tmp:]
        y_train = y[:tmp]
        y_val = y[tmp:]

        del x, y, perm

        """ Callback """

        # callbacks = [ReduceLROnPlateau(monitor='categorical_accuracy', patience=3)]

        # ModelCheckpoint(file_path, monitor='val_categorical_accuracy', save_best_only=True)

        datagen = ImageDataGenerator(rotation_range=180, horizontal_flip=True, vertical_flip=True, shear_range=0.2,
                                     height_shift_range=0.2, width_shift_range=0.2,
                                     dtype='float32')
        datagen.fit(x_train)

        """ Training loop """

        t0 = time.time()

        best_acc = 1e-10
        for epoch in range(nb_epoch):
            t1 = time.time()
            print("### Model Fitting.. ###")
            print('epoch = {} / {}'.format(epoch + 1, nb_epoch))
            print('check point = {}'.format(epoch))

            hist = model.fit_generator(datagen.flow(x_train, y_train, batch_size=batch_size),
                                       steps_per_epoch=x_train.shape[0] // batch_size,
                                       validation_data=(x_val, y_val))

            t2 = time.time()
            print(hist.history)
            print('Training time for one epoch : %.1f' % ((t2 - t1)))
            train_acc = hist.history['categorical_accuracy'][0]
            train_loss = hist.history['loss'][0]
            val_acc = hist.history['val_categorical_accuracy'][0]
            val_loss = hist.history['val_loss'][0]

            # must??
            best_model = (val_acc > best_acc)
            if best_model:
                nsml.save('best')
                best_acc = val_acc
            nsml.report(summary=True, step=epoch, epoch_total=nb_epoch, loss=train_loss, acc=train_acc,
                        val_loss=val_loss, val_acc=val_acc)
            nsml.save(epoch)
        print('Total training time : %.1f' % (time.time() - t0))
