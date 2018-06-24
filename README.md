# facialexpression
表情の画像認識についてのプログラムをまとめたファイル。

fitting.py
→顔の特徴量抽出のプログラム

CKplus.py
→cohnkanadeの画像(800枚)について、真顔と変化後の顔の各特徴量の差分をとり、データをまとめたもの

CK+test.py
→CKPlus.pyによって書き込まれたデータについて機械学習(randomforest)を実行するためのプログラム

PCAtest.py
→画像800枚についての主成分分析を行ったプログラム
