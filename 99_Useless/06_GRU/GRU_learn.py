import os
import csv
import tqdm
import keras
import pandas as pd
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Dropout, SimpleRNN, GRU, LSTM
from keras.optimizers import Adam

from matplotlib import pyplot as plt


if __name__ == '__main__':

    '''
    load csv file
    '''
    st = pd.read_csv('039.csv', header = None)
    dn = pd.read_csv('040.csv', header = None)

    print(st.info)
    print(dn.info)

    # count lines
    # st_lines = len(st)
    # dn_lines = len(dn)

    # get col[n]: xx.loc[n]
    # ex. 'stationary' line 3000: st.loc[3000]

    '''
    Create learning and test dataset
    '''
    class_number = 2
    data_num = len(st) + len(dn)

    print(data_num)

    data = []
    label = []

    for i in tqdm.tqdm(range(data_num)):
        if i % 2 == 0:
            data.append(st.loc[i/2])
            label.append('0')
        else:
            data.append(dn.loc[(i-1)/2])
            label.append('1')

    # for i in tqdm.tqdm(range(len(dn))):
    #     data.append(dn.loc[i])
    #     label.append('1')

    data = np.array(data)
    label = np.array(label)
    print(data.shape)
    print(label.shape)

    data = data.astype('float32')
    label = keras.utils.to_categorical(label, class_number)

    # data = data.reshape(data.shape[0], -1)
    # print(data.shape)

    '''
    model construction
    '''
    model = Sequential()
    model.add(GRU(32, input_shape = (50, 1)))
    model.add(Dropout(0.2))
    model.add(Dense(128, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(512, activation = 'relu'))
    model.add(Dropout(0.2))
    model.add(Dense(class_number, activation = 'softmax'))
    model.summary()

    model.compile(loss = 'categorical_crossentropy', optimizer = Adam(), metrics = ['accuracy'])

    history = model.fit(data, label, batch_size=None, epochs=5, verbose=1, validation_split=0.2)

    '''
    model evaluate
    '''
    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('Model accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.grid()
    plt.legend(['Train', 'Validation'], loc='upper left')
    plt.show()

    '''
    learning model save
    '''
    # open('gru_model.json', 'w').write(model.to_json())
    # model.save_weights('gru_model.weights.h5')
    model.save('gru_model.h5')
