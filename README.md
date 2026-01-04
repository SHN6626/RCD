
# Regularized Contrastive Decoding with Hard Negative Samples for LLM Hallucination Mitigation

This repository contains the official code for EMNLP 2025 Findings paper ["Regularized Contrastive Decoding with Hard Negative Samples for LLM Hallucination Mitigation"]((https://aclanthology.org/2025.findings-emnlp.322/#)).

## Introduction

Despite their impressive capabilities, large language models (LLMs) have been observed to generate responses that include inaccurate or fabricated information, a phenomenon commonly known as ``hallucination''. In this work, we propose a simple Induce-then-Contrast Decoding (ICD) strategy to mitigate this issue. We first construct a factually weak LLM by inducing hallucinations from the original LLMs. Then, we penalize these induced hallucinations during decoding to enhance the factuality of the generated content. Concretely, we determine the final next-token predictions by amplifying the predictions from the original model and downplaying the induced untruthful predictions via contrastive decoding. Experimental results on both discrimination-based and generation-based hallucination evaluation benchmarks, such as TruthfulQA and FActScore, demonstrate that our proposed ICD methods can effectively enhance the factuality of LLMs across various model sizes and families. For example, when equipped with our approach, Llama2-7B-Chat and Mistral-7B-Instruct now can achieve performance comparable to ChatGPT and GPT4 on TruthfulQA, respectively.

On TruthfulQA, our ICD approach significantly improves the truthfulness of Llama2-7B-Chat (+8 MC1 score) and Mistral-7B-Instruct (+20 MC1 score). With these improvements, the enhanced Llama2-7B-Chat and Mistral-7B-Instruct now match the performance levels of ChatGPT and GPT4, respectively.

## How to Install

You can use the following commands to install the environment for RCD:

```sh
conda create -n rcd python==3.10
conda activate rcd
pip install -r requirements.txt
cd ./transformers
pip install --editable ./
```

## Run

Try the following command to test our method on TruthfulQA:
```sh
cd ./exp_scripts/benchmark
sh truthfulqa.sh
```

For experiments on Factor, please try:
```sh
cd ./exp_scripts/benchmark
sh factor.sh
```

For experiments on Triviaqa and Natural Questions, please try:
```sh
cd ./exp_scripts/benchmark
sh know_seek.sh
```

We also provide some hallucinated models on the huggingface model hub for fast trial:
| Model | Link |
| :------- | :---------: |
| **kabumm/halu_llama_qa** | [HuggingFace](https://huggingface.co/kabumm/halu_llama_qa))|
| **kabumm/halu_llama_dia** | [HuggingFace](https://huggingface.co/kabumm/halu_llama_dia)|
| **kabumm/halu_llama_sum** | [HuggingFace](https://huggingface.co/kabumm/halu_llama_sum) |

## Contact

If you have any questions, please feel free to [email](mailto:shenghaonan@iie.ac.cn) me or drop me an issue.
