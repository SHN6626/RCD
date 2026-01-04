
# Regularized Contrastive Decoding with Hard Negative Samples for LLM Hallucination Mitigation

This repository contains the official code for EMNLP 2025 Findings paper ["Regularized Contrastive Decoding with Hard Negative Samples for LLM Hallucination Mitigation"](https://aclanthology.org/2025.findings-emnlp.322/#).

## Introduction

Large language models are prone to generate hallucinations, which can undermine their reliability in high-stakes applications. Some works on LLM hallucination mitigation use the model’s internal signals to contrast different output during inference stage. However, these works often focus on simple forms of hallucinations, and struggle to effectively mitigate hallucinations. To address the issue, this paper exploits hard negative samples to construct a factually weaker model for improving contrastive decoding. We propose a new inference-time method, Regularized Contrastive Decoding (RCD), to capture correct hallucination signals for mitigating hallucinations in LLMs. RCD learns more diverse hallucination patterns via adversarial-aware fine-tuning and mitigates hallucinations via contrastive decoding. Experiments on four hallucination benchmarks demonstrate that our method achieves better LLM hallucination mitigation performance. Further analysis shows RCD generalizes well across different model sizes, task formats, perturbation methods and training data sizes.

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
| **kabumm/halu_llama_qa** | [HuggingFace](https://huggingface.co/kabumm/halu_llama_qa)|
| **kabumm/halu_llama_dia** | [HuggingFace](https://huggingface.co/kabumm/halu_llama_dia)|
| **kabumm/halu_llama_sum** | [HuggingFace](https://huggingface.co/kabumm/halu_llama_sum) |

## Contact

If you have any questions, please feel free to [email](mailto:shenghaonan@iie.ac.cn) me or drop me an issue.
