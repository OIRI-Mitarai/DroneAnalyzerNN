import tqdm
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Input, LSTM, Dense, Attention, Dropout

from matplotlib import pyplot as plt

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
seq_length = 63

'''
create learning dataset
'''
# load csv
normal_train_data = load_data('039_all.csv')
abnormal_train_data = load_data('040_all.csv')

# create sequence
X_train_normal, y_train_normal = create_sequences(normal_train_data, seq_length)
X_train_abnormal, y_train_abnormal = create_sequences(abnormal_train_data, seq_length)

# combine data - normal and abnormal
X_train = np.concatenate([X_train_normal, X_train_abnormal], axis=0)
y_train = np.concatenate([np.zeros(len(X_train_normal)), np.ones(len(X_train_abnormal))], axis=0)

# display setting
print('seq_length:' + str(X_train.shape[1]))
print('features:' + str(X_train.shape[2]))

'''
model construction
'''
# model = Sequential()
# model.add(LSTM(units=64, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])))
# model.add(Attention())
# model.add(LSTM(units=32, return_sequences=False))  # 最後のLSTM層でシーケンスをフラットにする
# model.add(Dense(1, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam')

# input layer
inputs = Input(shape=(X_train.shape[1], X_train.shape[2]))

# hidden layer - LSTM
x = LSTM(units=128, return_sequences=True)(inputs)
x = LSTM(units=256, return_sequences=True)(x)

# Attention layer (Self-Attention)
attention_output = Attention()([x, x])
'''
# multihead attention
from keras.layers import MultiHeadAttention, LayerNormalization
attn_output = MultiHeadAttention(num_heads=8, key_dim=128)(x, x)
x = LayerNormalization(epsilon=1e-6)(attn_output +
'''

# last LSTM layer
x = LSTM(units=128, return_sequences=False)(attention_output)
x = Dropout(0.5)(x)

# output layer
outputs = Dense(1, activation='sigmoid')(x)

# model compile
model = Model(inputs=inputs, outputs=outputs)
model.summary()
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# fitting
# history = model.fit(X_train, y_train, batch_size=None, epochs=5, verbose=1, validation_split=0.2)
history = model.fit(X_train, y_train, batch_size=None, epochs=5, verbose=1)

# save attention model
model.save('Attention.h5')

'''
model evaluate
'''
plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.grid()
plt.legend(['Train'], loc='upper left')
plt.show()

'''
# load csv for learning
normal_test_data = load_data('022.csv')
abnormal_test_data = load_data('021.csv')
# prediction for normal data
X_test_normal = create_sequences(normal_test_data, seq_length)
y_pred_normal = model.predict(X_test_normal)

# prediction for abnormal data
X_test_abnormal = create_sequences(abnormal_test_data, seq_length)
y_pred_abnormal = model.predict(X_test_abnormal)

# evaluation
y_test_normal = np.zeros(len(X_test_normal))
y_test_abnormal = np.ones(len(X_test_abnormal))
y_test = np.concatenate([y_test_normal, y_test_abnormal])
y_pred = np.concatenate([y_pred_normal, y_pred_abnormal])

# ROC AUC
roc_auc = roc_auc_score(y_test, y_pred)
print('ROC AUC:', roc_auc)

# precision(適合率), recall(再現率), F-Measure(F1値)
print(classification_report(y_test, np.round(y_pred)))
'''
