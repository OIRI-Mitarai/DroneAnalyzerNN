import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

'''
load csv file function
'''
def load_data(file_path):
    data = pd.read_csv(file_path, header = None)
    # TODO - preprocessing if it needed
    return data

'''
load csv file
'''
data0 = load_data('039.csv')

# データを結合
'''
combine all data
'''
# X = np.concatenate([data1.values, data2.values, data0.values], axis=0)
# y = np.array([1] * len(data1) + [2] * len(data2) + [0] * len(data0))
X = np.concatenate([data0.values], axis=0)
y = np.array([0] * len(data0))

'''
construst failure detection model
'''
clf = IsolationForest(contamination=0.001)  # specify failure data rate
clf.fit(X)

'''
model evaluate
'''
y_pred = clf.predict(X)
y_scores = clf.decision_function(X)

'''
save learning model
'''
with open('isolation_forest_model.pkl', 'wb') as f:
    pickle.dump(clf, f)

'''
test data evaluate
'''
# load failure data
df_abnormal = pd.read_csv('021.csv')
# df_abnormal = pd.read_csv('023.csv')
# df_abnormal = pd.read_csv('026.csv')
# df_abnormal = pd.read_csv('028.csv')
# df_abnormal = pd.read_csv('030.csv')

# load normal
df_normal = pd.read_csv('022.csv')
# df_normal = pd.read_csv('024.csv')
# df_normal = pd.read_csv('025.csv')
# df_normal = pd.read_csv('027.csv')
# df_normal = pd.read_csv('029.csv')

# test data prediction
y_pred_abnormal = clf.predict(df_abnormal)
y_pred_normal = clf.predict(df_normal)

# confusion matrix for failure data
conf_matrix_abnormal = confusion_matrix(y_true=[-1]*len(y_pred_abnormal), y_pred=y_pred_abnormal)
print(conf_matrix_abnormal)

# confusion matrix for normal data
conf_matrix_normal = confusion_matrix(y_true=[1]*len(y_pred_normal), y_pred=y_pred_normal)
print(conf_matrix_normal)

# evaluation for all data
y_true = np.concatenate([[-1]*len(y_pred_abnormal), [1]*len(y_pred_normal)])
y_pred = np.concatenate([y_pred_abnormal, y_pred_normal])
print(classification_report(y_true, y_pred))

'''
# 可視化
import seaborn as sns
# 次元削減 (PCAを使用)
pca = PCA(n_components=3)
X_reduced = pca.fit_transform(X)

# データフレームに結果を格納
df = pd.DataFrame(X_reduced, columns=['PC1', 'PC2', 'PC3'])
df['anomaly'] = y_pred
df['score'] = y_scores

# 散布図の作成
sns.scatterplot(data=df, x='PC1', y='PC2', hue='anomaly', palette='coolwarm')
plt.title('Isolation Forest Anomaly Detection')
plt.show()
'''
