from keras.datasets import cifar10
from keras.models import model_from_json
from keras.utils.np_utils import to_categorical
from keras.optimizers import Adam

import numpy as np
import matplotlib.pyplot as plt

from keras import backend as K

batch_size = 32
epochs = 5

eps = 0.01
loop = 2
max_n = 100

num_classes = 10

def predict(x, model):
    pred = model.predict(np.array([x]), batch_size=1)
    pred_class = np.argmax(pred)
    pred_per = max(pred[0])

    return pred_class, pred_per

def generate_p(x, label, model):
    class_output = model.output[:, int(label)]

    grads = K.gradients(class_output, model.input)[0]
    gradient_function = K.function([model.input], [grads])

    grads_val = gradient_function([np.array([x])])

    p = np.sign(grads_val)

    return p.reshape(32,32,3)

def generate_adv(x, label, model, eps):
    p = generate_p(x, label, model)
    adv = (x - eps*p).clip(min=0, max=1)

    return adv

def generate_adv_list(x_list, y_list, model, eps):
    adv_list = []

    for x, y in zip(x_list, y_list):
        adv = generate_adv(x, y, model, eps)
        adv_list.append(adv)

    return np.array(adv_list)

# データの読み込み
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# 正規化
x_train = x_train.astype(np.float32) / 255.0
x_test = x_test.astype(np.float32) / 255.0

y_train_catego = to_categorical(y_train, num_classes)
y_test_catego = to_categorical(y_test, num_classes)

print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# モデルを読み込む
model = model_from_json(open('models/10_model.json').read())

# 学習結果を読み込む
model.load_weights('weights/10_weights.h5')

model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=0.001), metrics=['accuracy'])

print(model.summary())

score = model.evaluate(x_test, y_test_catego, verbose=0)
print('x_test')
print('Test loss :', score[0])
print('Test accuracy :', score[1])

label =['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'boat', 'track']

adv_test = []
for i in range(loop):
    advs = generate_adv_list(x_test[i*max_n:(i+1)*max_n], y_test[i*max_n:(i+1)*max_n], model, eps)
    adv_test.extend(advs)
    print(i)

adv_test = np.array(adv_test)

score = model.evaluate(x_test[:loop*max_n], y_test_catego[:loop*max_n], verbose=0)
print('x_test', loop*max_n)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

score = model.evaluate(adv_test, y_test_catego[:loop*max_n], verbose=0)
print('adv_test', loop*max_n)
print('Test loss :', score[0])
print('Test accuracy :', score[1])


adv_train = []
for i in range(loop):
    advs = generate_adv_list(x_train[i*max_n:(i+1)*max_n], y_train[i*max_n:(i+1)*max_n], model, eps)
    adv_train.extend(advs)
    print(i)

adv_train = np.array(adv_train)

history = model.fit(adv_train, y_train_catego[:loop*max_n],
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=1,
                    validation_data=(x_test, y_test_catego))

score = model.evaluate(x_test, y_test_catego, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

model_json_str = model.to_json()
open('models/200advs_model.json', 'w').write(model_json_str)
model.save_weights('weights/200advs_weights.h5');
print('model saved.')

score = model.evaluate(x_test, y_test_catego, verbose=0)
print('x_test')
print('Test loss :', score[0])
print('Test accuracy :', score[1])

score = model.evaluate(x_test[:loop*max_n], y_test_catego[:loop*max_n], verbose=0)
print('x_test', loop*,max_n)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

score = model.evaluate(adv_test, y_test_catego[:loop*max_n], verbose=0)
print('adv_test', loop*max_n)
print('Test loss :', score[0])
print('Test accuracy :', score[1])

# 正解率の推移をプロット
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Accuracy')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig("graphs/200adv_acc.png")

plt.cla()

# ロスの推移をプロット
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Loss')
plt.legend(['train', 'test'], loc='upper left')
# plt.show()
plt.savefig("graphs/200adv_loss.png")
