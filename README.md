# Anomaly sound detector
本ツールはあくまで実験用途として作成したツール群となります  
OSSとして公開しますが、弊社（JSP）にてサポート等は一切行いませんので、ご了承戴いた上でお使いください。  

## License
### モデル定義部
MIT License  
Copyright (c) 2020 Hitachi, Ltd.

### 上記以外
GPLv3 License  
Copyright (c) 2021 JSP Co., Ltd.


## インストール方法
下記コマンドでインストールしてください

```shell
python setup.py install
```

## 使い方

### データ準備

#### 1. 学習データ
学習対象とする WAVファイルを in フォルダ直下に配置し、下記コマンドを入力してください
```shell
adpreproc
```
* 入力：　in/*.wav　　学習対象WAVファイル（フォルダ内全WAVファイルが対象となります）
* 出力：　in/feature/*.csv　　特徴量ファイル

#### 2. 評価データ
学習対象とする WAVファイルから1ファイルを選択し、下記コマンドを入力してください  
```shell
admixer (input_wav_file)
```
* 入力：　in/xxx.wav　　学習対象WAVファイルから1ファイルを選択してください
* 出力：　in/sample/eval_*.wav　　評価音（正常音・異常音）

### 学習
"adpreproc" で学習データを作成した後、下記コマンドを入力してください
```shell
adtrainer
```
* 入力：　in/feature/*.csv　　特徴量ファイル
* 出力：　out/ckpt/*　　学習済みモデル

### 推論
"admixer" で評価データ作成、及び "adtrainer" で学習後、下記コマンドを入力してください
```shell
adevaluator
```
* 入力：　in/sample/eval_*.wav　　評価音（正常音・異常音）
* 入力：　out/ckpt/*　　学習済みモデル
* 出力：　out/result/*.csv　　結果CSVファイル

