import os
import csv
import tqdm
import keras

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


from keras.models import load_model
from keras.utils import to_categorical

if __name__ == '__main__':

    '''
    load model
    '''
    model = load_model('gru_model.h5')

    '''
    load csv file
    '''
    print('==========================================')
    csv_file = input('Enter the CSV file to be analyzed:')
    test_csv = pd.read_csv('data/' + csv_file, header = None)
    print(test_csv.info)

    '''
    create test dataset
    '''
    class_number = 2
    data_num = len(test_csv)

    data = []
    # label = []

    for i in tqdm.tqdm(range(data_num)):
        data.append(test_csv.loc[i])
        # label.append('')

    data = np.array(data)
    # label = np.array(data)
    print(data.shape)

    data = data.astype('float32')
    # label = keras.utils.to_categorical(label, class_number)

    '''
    test
    '''
    # print('Now analyzing....')
    pred = model.predict(data, verbose = 1)
    results = pred.argmax(axis = 1)

    rate = np.arange(data_num, dtype = float)
    for i in tqdm.tqdm(range(data_num)):
        rate[i] = np.array(pred[i][1])

    plt.plot(range(data_num), rate, linestyle = 'solid')
    plt.title('Moter Stop Probability')
    plt.ylabel('Rate')
    plt.xlabel('Time[ms]')
    plt.ylim(-0.1, 1.1)
    plt.grid()
    plt.show()
