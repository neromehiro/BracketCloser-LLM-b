# BracketCloser-LLM プロジェクト

BracketCloser-LLMは、大規模言語モデル（LLM）について学ぶためのプロジェクトで、簡単なLLMを作成し、実際にLLMが学習できることを体験します。限られたトークンセットを使用しており、簡単にルールを学習できます。

## プロジェクト概要

- **目的**: 簡単なLLMを作成し、LLMの学習プロセスを理解する。
- **トークン**: 以下の9つのトークンを使用します。
  ```python
  tokens = ["(", ")", "[", "]", "{", "}", "input", ",output", ","]
  ```
- **学習タスク**: 括弧を閉じるルールを学習します。
- **アーキテクチャ**: いくつかの最新のアーキテクチャ（GRU、Transformer、LSTM、BERT、GPT）を通じて、LLMの作成方法を学びます。

## 使い方

### ステップ 1: Docker イメージの取得

```sh
docker pull nero1014/bracket-closer-image
```

### ステップ 2: Docker コンテナのセットアップと実行

```sh
export PROJECT_DIR=$(pwd)
docker run -it \
  -v "${PROJECT_DIR}:/app/project" \
  --workdir /app/project \
  --tmpfs /tmp:rw,size=10g \
  bracket-closer-image \
  bash
```

### ステップ 3: データセットの作成

以下のコードを実行してデータセットを生成します。生成するサンプル数を調整するには、`num_samples`の値を変更します。最適な学習時間のために、300〜500サンプルを推奨します。

```sh
python dataset.py
```

### ステップ 4: モデルの学習

学習スクリプトを実行します。

```sh
python train.py
```

表示されるリストから学習したいモデルを選びます。以下のモデルから選択できます（例: "gru 1min"）。

- **GRU（Gated Recurrent Unit）**
  - 1分: `gru 1min`
  - 10分: `gru 10min`
  - 1時間: `gru 1hour`
  - 6時間: `gru 6hours`
- **Transformer**
  - 1分: `tra 1min`
  - 10分: `tra 10min`
  - 1時間: `tra 1hour`
- **LSTM**
  - 1分: `lstm 1min`
  - 10分: `lstm 10min`
  - 1時間: `lstm 1hour`
- **BERT**
  - 1分: `ber 1min`
  - 10分: `ber 10min`
- **GPT**
  - 1分: `gpt 1min`

学習されたモデルは `models` フォルダに保存されます。例えば、`gru_20240605_163409_15m`というフォルダ名は、GRUモデルが2024年6月5日16時34分9秒に学習され、15分かかったことを示しています。

### ステップ 5: モデルの評価

以下のコマンドでモデルを評価します。

```sh
python evaluate.py
```

最新順に並んだモデルのリストが表示されます。評価したいモデルを番号で選択します。各評価は100個のテストデータを生成し、正解率がパーセンテージで表示されます。

特定のモデルを評価するには、以下を実行します。

```sh
python evaluate2.py
```

評価したいモデルの相対パスを入力します。

### ステップ 6: Optuna を使ったハイパーパラメーターチューニング

以下のコマンドでハイパーパラメーターチューニングを行います。

```sh
python hyper.py
```

過去の学習を継続するか、新規で学習を始めるかを選べます。モデルアーキテクチャ、学習時間、並列ジョブ数（1〜5推奨）を入力します。

例:

```sh
ubuntu@8326ea14da18:/app/project$ python hyper.py
Choose an option:
1. Resume existing study
2. Start a new study
Enter 1 or 2: 2
Enter the model architecture (gru, transformer, lstm, bert, gpt): gru
Enter the training time limit (e.g., '3min', '1hour', '5hour'): 3min
Optimization Progress:   0%|             | 0/180.0 [00:00<?, ?s/s]Enter the number of parallel jobs: 1
[I 2024-06-06 11:08:39,182] A new study created in RDB with name: hyper_gru_3
```

モデルは `optuna_studies` フォルダに保存されます。`evaluate2.py` を使ってそのモデルを評価します。

## 学習と評価の例

LLMは入力シーケンスに基づいて括弧を閉じる予測を学習します。以下はその例です：

```python
original = [
    "input:(){({}){【】【】(){}{,output:}}}",
    "input:{【{}【】【】()】{{}},output:}",
]

preprocessed_sequence = [
    [0, 1, 4, 0, 4, 5, 1, 4, 2, 3, 2, 3, 0, 1, 4, 5, 4, 8, 5, 5, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [4, 2, 4, 5, 2, 3, 2, 3, 0, 1, 3, 4, 4, 5, 5, 8, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [2, 2, 3, 2, 2, 3, 2, 3, 0, 4, 5, 4, 5, 4, 5, 1, 8, 3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
]
```

### 評価の例

モデルの入力に `input` と `output` を含めた際の評価例を以下に示します。

#### 出力例 (evaluation_result.txt)：

- **問題1**: 不正解
  - 入力: `input:【】()【{{}}】({(),output`
  - 出力: `((`
  - 正解: `})`
  
- **問題2**: 不正解
  - 入力: `input:【【({{}【】}【【】】)】,output`
  - 出力: `(`
  - 正解: `】`
  
- **問題3**: 不正解
  - 入力: `input:(【】(())(,output`
  - 出力: `((`
  - 正解: `))`

BracketCloser-LLMの機能を探求し、実際にLLMの学習プロセスを体験してください！
