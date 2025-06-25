import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.layers import LSTM, Dense, Input, RepeatVector, Lambda


import sys
sys.stdout.reconfigure(encoding='utf-8')



print('=============================')
print('Encoder-Decoder_learn START')
print('=============================')

# 設定
DATA_DIR = "train_light"
MODEL_NAME = "model/EncoderDecoder.keras"  # .keras 形式に変更
CHECKPOINT_NAME = "model/EncoderDecoder_checkpoint.keras"  # .keras 形式に変更
FEATURES = 50
LABELS = 12
SEQUENCE_LENGTH = 100
EPOCHS = 20
BATCH_SIZE = 64

# ディレクトリが存在しない場合は作成
os.makedirs(os.path.dirname(MODEL_NAME), exist_ok=True)

# CSV読み込み関数
def load_data(files):
    X, y = [], []
    for file in files:
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        data = df.iloc[:, 1:1+FEATURES].values
        labels = df.iloc[:, 1+FEATURES].values
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i:i + SEQUENCE_LENGTH])
            y.append(labels[i + SEQUENCE_LENGTH])
    return np.array(X), np.array(y)

# trainフォルダ内のCSVを取得
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
print(f"Loaded files: {csv_files}")

# データ読み込み
X_train, y_train = load_data(csv_files)
print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")

# モデル構築
def build_model():
    encoder_inputs = Input(shape=(SEQUENCE_LENGTH, FEATURES))
    encoder = LSTM(128, return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)

    decoder_inputs = RepeatVector(SEQUENCE_LENGTH)(encoder_outputs)
    decoder = LSTM(128, return_sequences=True)
    decoder_outputs = decoder(decoder_inputs, initial_state=[state_h, state_c])

    outputs = Lambda(lambda x: x[:, -1, :])(decoder_outputs)
    outputs = Dense(LABELS, activation="softmax")(outputs)

    model = Model(encoder_inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model

# モデル構築
model = build_model()

# コールバック設定
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=1)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)
checkpoint = tf.keras.callbacks.ModelCheckpoint(CHECKPOINT_NAME, save_best_only=True, save_weights_only=False, verbose=1)

# 学習
try:
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[reduce_lr, early_stopping, checkpoint],
        verbose=1
    )
    model.save(MODEL_NAME)  # 学習が完了したら保存
except KeyboardInterrupt:
    print("\n学習が中断されました。最新のモデルを保存します。")
    model.save(MODEL_NAME)

print('====================================')
print('Encoder-Decoder_learn COMPLETED!!!')
print('====================================')
