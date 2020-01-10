import numpy as np

path = 'npy/'

first = np.load(path + 'advs0_5000_train.npy')
second = np.load(path + 'advs5000_10000_train.npy')
third = np.load(path + 'advs10000_15000_train.npy')
four = np.load(path + 'advs15000_20000_train.npy')
five = np.load(path + 'advs20000_25000_train.npy')
six = np.load(path + 'advs25000_30000_train.npy')
seven = np.load(path + 'advs30000_35000_train.npy')
eight = np.load(path + 'advs35000_40000_train.npy')
nine = np.load(path + 'advs40000_45000_train.npy')
ten = np.load(path + 'advs45000_50000_train.npy')

train = np.concatenate([first, second, third, four, five, six, seven, eight, nine, ten])
print(train.shape)

np.save(path + 'advs_train_50_model.npy', train)
print('Saved')
