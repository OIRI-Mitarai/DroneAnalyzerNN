import os
import gc
import tqdm
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

# データ読み込み関数
def load_data(filename):
    data = pd.read_csv(filename, header=None)
    return data.values

# シーケンス作成関数
def create_sequences(data, seq_length):
    X, y = [], []
    for i in tqdm.tqdm(range(len(data) - seq_length)):
        seq = data[i:(i + seq_length), :]
        label = data[i + seq_length, 0]
        X.append(seq)
        y.append(label)
    return np.array(X), np.array(y)

# シーケンス長とクラス数
seq_length = 100
class_num = 12

# 学習済みモデルの読み込み
model = load_model(r'./model/EncoderDecoder.keras', safe_mode=False)

# 対象のCSVファイルリスト
DATA_DIR = "test"
csv_files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith(".csv")])

# 各CSVファイルに対して処理を実行
for csv_file in csv_files:
    # テストデータの読み込みと整形
    test_data = load_data(r'./test/' + csv_file)
    X_test, _ = create_sequences(test_data, seq_length)

    # 予測の実行
    y_pred_probs = model.predict(X_test)
    y_pred = np.argmax(y_pred_probs, axis=1)
    confidence = np.max(y_pred_probs, axis=1)

    # 結果の数値化
    print(f"予測結果 for {csv_file}:")
    print(f"y_pred: {y_pred}")
    print(f"confidence: {confidence}")

    # 結果をCSVファイルに保存
    result_df = pd.DataFrame(y_pred_probs, columns=[f"Class_{i}" for i in range(class_num)])
    result_df.insert(0, "Predicted Class", y_pred)  # 予測クラスを最初の列に追加
    result_df.insert(1, "Max Confidence", confidence)  # 最大確信度を追加
    output_filename = f'results/Encoder-Decoder_prediction_with_all_confidence_{csv_file.split(".")[0]}.csv'
    result_df.to_csv(output_filename, index=False)

    # 結果のグラフ化と保存
    plt.figure(figsize=(12, 6))
    plt.plot(y_pred, marker='o')
    plt.title(f'Prediction Results for {csv_file} (Encoder-Decoder)')
    plt.xlabel('Time Step')
    plt.ylabel('Predicted Class')
    plt.yticks(range(class_num))
    plt.grid(True)
    output_filename = f'results/test_prediction_Encoder-Decoder_{csv_file.split(".")[0]}.png'
    plt.savefig(output_filename)

    # 🆕 メモリ解放（大きなNumpy配列を削除）
    del X_test, y_pred_probs, y_pred, confidence, result_df
    gc.collect()  # ガベージコレクションを実行


    # 🆕 Matplotlib のキャッシュを完全解放
    plt.close('all')
    gc.collect()  # 追加のメモリ解放

    print(f"Finished processing {csv_file}")


print("All files processed successfully!")