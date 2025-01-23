import tqdm
import keras

import numpy as np
import pandas as pd

from matplotlib import pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, roc_auc_score

from keras.models import load_model


# load data function
def load_data(filename):
    data = pd.read_csv(filename, header=None)
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data)
    return scaled_data

# create sequences function
def create_sequences(data, seq_length):
    X, y = [], []
    for i in tqdm.tqdm(range(len(data)-seq_length)):
        seq = data[i:(i+seq_length), :]
        label = data[i+seq_length, 0]  # 最初の列を予測対象とする
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)


'''
devide every xx[ms] data
xx[ms] is one sentence
'''
seq_length = 50

'''
load model
'''
model = load_model('Attention.h5')

'''
create test dataset
'''
# load csv for learning
# normal_test_data = load_data('022.csv')
# abnormal_test_data = load_data('021.csv')
normal_csv = input('Normal flight csv data:')
normal_test_data = load_data(normal_csv)
abnormal_csv = input('Abnormal flight csv data:')
abnormal_test_data = load_data(abnormal_csv)
# prediction for normal data
X_test_normal, y_test_normal_tmp = create_sequences(normal_test_data, seq_length)
y_pred_normal = model.predict(X_test_normal)

'''
prediction
'''
# prediction for abnormal data
X_test_abnormal, y_test_abnormal_tmp = create_sequences(abnormal_test_data, seq_length)
y_pred_abnormal = model.predict(X_test_abnormal)

# evaluation
y_test_normal = np.zeros(len(X_test_normal))
y_test_abnormal = np.ones(len(X_test_abnormal))
y_test = np.concatenate([y_test_normal, y_test_abnormal])
y_pred = np.concatenate([y_pred_normal, y_pred_abnormal])

'''
evaluation
'''
# ROC AUC
roc_auc = roc_auc_score(y_test, y_pred)
print('ROC AUC:', roc_auc)

# precision(適合率), recall(再現率), F-Measure(F1値)
print(classification_report(y_test, np.round(y_pred)))
