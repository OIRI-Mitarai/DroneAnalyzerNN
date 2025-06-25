import os
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import LSTM, Dense, Input, Attention, Add
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

print('======================')
print('Attention_learn START')
print('======================')

# 設定
DATA_DIR = "train_light"  # 学習データのフォルダ
MODEL_NAME = "model/Attention.h5"  # モデル保存名
CHECKPOINT_NAME = "model/Attention_checkpoint.weights.h5" # チェックポイントの保存名を追加
FEATURES = 50  # 特徴量の数
LABELS = 12  # クラス数
SEQUENCE_LENGTH = 100  # タイムステップ数
EPOCHS = 20  # 学習回数
BATCH_SIZE = 64  # バッチサイズ

# CSV読み込み関数（時刻を保持するが特徴量には含めない）
def load_data(files):
    X, y = [], []  # 時刻データも保持
    for file in files:
        df = pd.read_csv(os.path.join(DATA_DIR, file), header=None)
        # timestamps.append(df.iloc[:, 0].values)  # 1列目（時刻データ）を保存
        data = df.iloc[:, 1:1+FEATURES].values  # 特徴量（時刻は除外）
        labels = df.iloc[:, 1+FEATURES].values  # ラベル

        # シーケンスデータ作成（時系列を考慮）
        for i in range(len(data) - SEQUENCE_LENGTH):
            X.append(data[i:i + SEQUENCE_LENGTH])  # 過去 SEQUENCE_LENGTH ステップ分
            y.append(labels[i + SEQUENCE_LENGTH])  # その次の時点のラベル

    # return np.array(X), np.array(y), timestamps  # 時刻データも返す
    return np.array(X), np.array(y)

# trainフォルダ内のすべてのCSVを取得
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])
print(f"Loaded files: {csv_files}")



# LSTM + Attention モデル構築
def build_model():
    inputs = Input(shape=(SEQUENCE_LENGTH, FEATURES))
    x = LSTM(128, return_sequences=True)(inputs)
    x = LSTM(64, return_sequences=True)(x)
    attention = Attention()([x, x])  # Self-Attention
    x = Add()([x, attention])  # LSTMの出力にAttentionを加える
    x = LSTM(32)(x)
    outputs = Dense(LABELS, activation="softmax")(x)
    model = Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    return model

# モデル構築
model = build_model()

# データ読み込み
# X_train, y_train, timestamps = load_data(csv_files)
X_train, y_train = load_data(csv_files)

# ReduceLROnPlateau の追加
reduce_lr = ReduceLROnPlateau(monitor='loss', factor=0.5, patience=2, verbose=1, min_lr=1e-6)

# ModelCheckpoint の追加
checkpoint = ModelCheckpoint(CHECKPOINT_NAME, save_best_only=True, save_weights_only=True, verbose=1)

# EarlyStopping追加
early_stopping = tf.keras.callbacks.EarlyStopping(monitor='loss', patience=3, restore_best_weights=True, verbose=1)

# 学習
try:
    history = model.fit(
        X_train, y_train,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        callbacks=[reduce_lr, early_stopping, checkpoint],
        verbose=1
    )
except KeyboardInterrupt:
    print("\n学習が中断されました。現在の重みを保存します。")

# モデル保存（中断されなかった場合のみ）
if 'history' in locals():
    model.save(MODEL_NAME)

# 学習曲線プロット
def plot_history(history):
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # plt.show()
    plt.savefig('Attention_training_plot.png', dpi=300)

# historyが存在する場合のみプロット
if 'history' in locals():
    plot_history(history)

print('=============================')
print('Attention_learn COMPLETED!!!')
print('=============================')
