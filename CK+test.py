# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd

# CSVファイルを読み込む
df = pd.read_csv('CKplus.csv', header=None)

# 全行，1列目以降を取り出してnumpy.arrayに直す場合
images=np.array(df.loc[:,1:])

# 画像データを配列にしたもの(numpy.ndarray型)
X = images[:,0:135]
# 正解ラベル
y = images[:,136]

# 訓練データとテストデータに分ける
# 訓練データ ：前半
X_train, y_train = X[0:300], y[0:300]
# テストデータ：後半
X_test, y_test = X[300:], y[300:]

from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
#clf = svm.SVC(gamma=0.001)
clf=RandomForestClassifier()

# 訓練データとラベルで学習
clf.fit(X_train, y_train)

# テストデータで試した正解率を返す
accuracy = clf.score(X_test, y_test)
print("accuracy="+str(accuracy))

# 学習済モデルを使ってテストデータを分類した結果を返す
predicted = clf.predict(X_test)

from sklearn.metrics import classification_report,confusion_matrix
# 詳しいレポート
# precision(適合率): 選択した正解＝/（０を）選択した集合＝嘘を付かない割合
# recall(再現率) : 選択した正解/全体の正解＝どれだけ再現できたか
# F-score(F値) : 適合率と再現率はトレードオフの関係にあるため
print("classification report")
print(classification_report(y_test, predicted))
print(confusion_matrix(y_test, predicted))
