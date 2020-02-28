from keras.datasets import cifar10
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Conv2D, MaxPooling2D, Flatten
from keras.optimizers import SGD

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K

batch_size = 128
epochs = 200

eps = 0.01

p = 0.5
np.random.seed(0)

num_classes = 10

model_name = '150'
# save_model_name = '200_advs001_p05'
# load_npy = path + 'npy/adv_train_150_001.npy'

label =['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

def load_model(model_name):
    model = model_from_json(open('models/' + model_name + '_model.json').read())
    model.load_weights('weights/' + model_name + '_weights.h5')

    model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

    return model

def predict(x, model_name):
    model = load_model(model_name)
    pred = model.predict(np.array([x]), batch_size=1)
    pred_class = np.argmax(pred)
    pred_per = max(pred[0])
    K.clear_session()

    return pred_class, pred_per

def generate_grads(x, label, model_name):

    model = load_model(model_name)

    class_output = model.output[:, int(label)]

    grads = K.gradients(class_output, model.input)[0]
    gradient_function = K.function([model.input], [grads])

    grads_val = gradient_function([np.array([x])])

    K.clear_session()

    return np.array(grads_val).reshape(32,32,3)

def generate_adv(x, label, model_name, eps):

    p = np.sign(generate_grads(x, label, model_name))
    adv = (x - eps*p).clip(min=0, max=1)

    return adv

def generate_adv_list(x_list, y_list, model_name, eps):
    adv_list = []

    for i, (x, y) in enumerate(zip(x_list, y_list)):
        if i % 100 == 0:
            print(f'{i}/{len(x_list)}')

        adv = generate_adv(x, y, model_name, eps)
        adv_list.append(adv)

    return np.array(adv_list)

def confuse_true_adv_list(x_list, adv_list, p):
    c = 0
    x_adv_list = np.copy(x_list)
    for i in range(x_adv_list.shape[0]):
        p1 = np.random.uniform(0,1)
        if p1 < p:
            c += 1
            x_adv_list[i] = adv_list[i]

    print(f'confuse : (adv) {c} / (all){x_adv_list.shape[0]}')
    return x_adv_list

# データの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# 正規化
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

y_train_catego = to_categorical(y_train, num_classes)
y_test_catego = to_categorical(y_test, num_classes)
#
# adv_test = generate_adv_list(x_test, y_test, model_name, eps)
# print(adv_test.shape)
# np.save('npy/adv_test_150_001.npy', adv_test)
#
# base_model = load_model(model_name)
# print(base_model.summary())
#
# score = base_model.evaluate(x_test, y_test_catego, verbose=0)
# print('x_test')
# print('Test loss :', score[0])
# print('Test accuracy :', score[1])
#
# score = base_model.evaluate(adv_test, y_test_catego, verbose=0)
# print('adv_test', test_n)
# print('Test loss :', score[0])
# print('Test accuracy :', score[1])

# print('x_test')
# plt.figure(figsize=(10,10))
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# for i in range(25):
#     cifar_img=plt.subplot(5,5,i+1)
#
#     img = x_test[i]
#     pred_class, pred_per = predict(img, model_name)
#     plt.imshow(img)
#     plt.title(f'{label[int(y_test[i])]} -> {label[pred_class]}')
#
# plt.show()
#
# print('adv_test')
# plt.figure(figsize=(10,10))
# plt.subplots_adjust(wspace=0.4, hspace=0.6)
# for i in range(25):
#     cifar_img=plt.subplot(5,5,i+1)
#
#     img = adv_test[i]
#     pred_class, pred_per = predict(img, model_name)
#     plt.imshow(img)
#     plt.title(f'{label[int(y_test[i])]} -> {label[pred_class]}')
#
# plt.show()

adv_train = generate_adv_list(x_train, y_train, model_name, eps)
print(adv_train.shape)
np.save('npy/150_adv01_train.npy', adv_train)
#
# adv_train = np.load(load_npy)
# print(adv_train.shape)
#
# x_adv_train = confuse_true_adv_list(x_train, adv_train, p)
# print(x_adv_train.shape)
#
# model = Sequential()
#
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(32,32,3)))
# model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
# model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
# model.add(MaxPooling2D(pool_size=(2, 2)))
# model.add(Dropout(0.25))
#
# model.add(Flatten())
# model.add(Dense(512))
# model.add(Activation('relu'))
# model.add(Dropout(0.5))
# model.add(Dense(10))
# model.add(Activation('softmax'))
#
# model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])
#
# print(model.summary())
#
# hist = model.fit(x_adv_train, y_train_catego,
#               batch_size=batch_size,
#               epochs=epochs,
#               verbose=1,
#               validation_split=0.1)
#
# model_json_str = model.to_json()
# open(save_model_name + '_model.json', 'w').write(model_json_str)
# model.save_weights(save_model_name + '_weights.h5')
# print('model saved')
#
# score = model.evaluate(x_test, y_test_catego, verbose=0)
# print('x_test')
# print('Test loss :', score[0])
# print('Test accuracy :', score[1])
#
# score = model.evaluate(x_test[:test_n], y_test_catego[:test_n], verbose=0)
# print('x_test', test_n)
# print('Test loss :', score[0])
# print('Test accuracy :', score[1])
#
# score = model.evaluate(adv_test, y_test_catego[:test_n], verbose=0)
# print('adv_test', test_n)
# print('Test loss :', score[0])
# print('Test accuracy :', score[1])
#
# # 正解率の推移をプロット
# plt.plot(hist.history['acc'])
# plt.plot(hist.history['val_acc'])
# plt.title('Accuracy')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# plt.cla()
#
# # ロスの推移をプロット
# plt.plot(hist.history['loss'])
# plt.plot(hist.history['val_loss'])
# plt.title('Loss')
# plt.legend(['train', 'test'], loc='upper left')
# plt.show()
#
# from sklearn.metrics import confusion_matrix
# predict_classes = model.predict_classes(x_test)
# true_classes = np.argmax(y_test_catego, 1)
# cmx = confusion_matrix(true_classes, predict_classes)
# print(cmx)
#
# import seaborn as sns
# sns.heatmap(cmx, annot=True, fmt='g', square=True)
# plt.show()
#
