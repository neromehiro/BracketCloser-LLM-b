# BracketCloser-LLM

最初にやること
docker pull sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1


export PROJECT_DIR=$(pwd)
docker run -it \
  -v "${PROJECT_DIR}:/app/project" \
  --workdir /app/project \
  --tmpfs /tmp:rw,size=10g \
  sonoisa/deep-learning-coding:pytorch1.12.0_tensorflow2.9.1 \
  bash



BracketCloser-LLMは、大規模言語モデル（LLM）について学ぶためのプロジェクトで、簡単なLLMを作成し、実際にLLMが学習できることを体験します。限られたトークンセットを使用しており、簡単にルールを学習できます。

## プロジェクト概要
- **目的**: LLMを学ぶために簡単なLLMを作成し、学習プロセスを理解する。
- **トークン**: LLMは以下の9つのトークンを使用します。
  ```
  tokens = ["(", ")", "[", "]", "{", "}", "input", ",output", ","]
  ```
- **学習タスク**: LLMは括弧を閉じるルールを学習します。
- **最新アーキテクチャ**: いくつかの最新のアーキテクチャ(gru,transformer,lstm,bert,gpt)を通じて、LLMの作成方法を学びます。

## 学習と評価の例
LLMは入力シーケンスに基づいて括弧を閉じる予測を学習します。以下はその例です：

```
Input: {}{}(){, Predicted Output: (, Expected Output: }
Input: {}()【】(){}【, Predicted Output: (, Expected Output: 】
Input: 【】{【{{}}{{}()【】, Predicted Output: (((, Expected Output: }】}
Input: (()【】【】)({, Predicted Output: ((, Expected Output: })
Input: 【()()【】【】【】, Predicted Output: (, Expected Output: 】
Input: (【{(()【】【】【】), Predicted Output: (((, Expected Output: }】)
Input: ()({{{()【】}}, Predicted Output: ((, Expected Output: })
Input: {【{}】(【】, Predicted Output: ((, Expected Output: )}
Input: {{}}(【, Predicted Output: ((, Expected Output: 】)
Input: (){【】{}【】【, Predicted Output: ((, Expected Output: 】}
Input: {{{}【】(), Predicted Output: ((, Expected Output: }}
Input: ((【{}()(【】)】){(), Predicted Output: ((, Expected Output: })
Input: (【{【】}, Predicted Output: ((, Expected Output: 】)
Input: {{(【】【】)}, Predicted Output: (, Expected Output: }
Input: ((()){【】【】()【】, Predicted Output: ((, Expected Output: )}
Input: 【】({((()()), Predicted Output: ()(, Expected Output: )})
Input: 【{{}, Predicted Output: }(, Expected Output: }】
Input: (()()【】【】【】{}【, Predicted Output: ((, Expected Output: 】)
Input: ((【({}{})(())()】), Predicted Output: (, Expected Output: )
Input: {({{}(【】{})}), Predicted Output: (, Expected Output: }
Input: ({}{}{, Predicted Output: ((, Expected Output: })
```