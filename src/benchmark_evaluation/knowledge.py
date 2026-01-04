import re
import os
import json
import numpy as np
import transformers
from tqdm import tqdm
import argparse
import sys
import ssl
import urllib.request
from datasets import load_dataset, load_from_disk
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, average_precision_score, auc, precision_recall_curve
import time
from pathlib import Path

from trivia_eval_util import evaluate_triviaqa, evaluate_nq
import ssl
import urllib.request
import zipfile
import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
DATA_DIR = file.parents[2] / "datasets"
sys.path.append(str(root))
transformers.logging.set_verbosity(40)
from decoding_algorithm import ContrastiveDecoding

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"


def load_dataset_data(dataset_name, debug):
    if dataset_name == 'triviaqa':
        dataset = load_from_disk("hotpotqa")['validation']
    elif dataset_name == 'natural_questions':
        dataset = load_from_disk("hotpotqa")['validation']
    elif dataset_name == 'hotpotqa':
        dataset = load_dataset("hotpotqa")['validation']
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented yet.")

    questions = list(dataset['question'])
    answers = list(dataset['answer'])

    if debug:
        questions = questions[:20]
        answers = answers[:20]

    return questions, answers


def build_prompt(question_text, prompt_style='zero_shot'):
    if prompt_style == 'zero_shot':
        prompt = 'Answer the following question concisely.\n'
        prompt += f'Q:{question_text}\nA:'
    elif prompt_style == 'few_shot':
        prompt = 'Answer the following question concisely.\n'
        prompt += 'Q: Who was President when the first Peanuts cartoon was published?\nA: Harry Truman\n\n'
        prompt += 'Q: Where in England was Dame Judi Dench born?\nA: York\n\n'
        prompt += f'Q: {question_text}\nA: '
    return prompt


def plot_auroc_scores(is_correct_list, scores_list, output_file, method_name):
    correct_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if is_correct]
    incorrect_scores = [score for is_correct, score in zip(is_correct_list, scores_list) if not is_correct]

    if np.isnan(correct_scores).any() or np.isnan(incorrect_scores).any():
        print(f"Error: there is nan, skip computing AUROC, AUPRC, AURC for {method_name}")
        scores = {'auroc': None, 'auprc': None, 'aurc': None}
        return scores

    y_true = [1] * len(correct_scores) + [0] * len(incorrect_scores)
    y_scores = correct_scores + incorrect_scores

    auroc = roc_auc_score(y_true, y_scores)
    auprc = average_precision_score(y_true, y_scores)
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    aurc = auc(recall, precision)

    plt.figure()
    plt.hist(correct_scores, bins=20, alpha=0.5, label='Correct')
    plt.hist(incorrect_scores, bins=20, alpha=0.5, label='Incorrect')
    plt.legend(loc='upper right')
    plt.title(f'AUROC: {auroc:.2f}')
    output_dir = os.path.dirname(output_file)
    plt.savefig(os.path.join(output_dir, f'detect_{method_name}_plot.png'))
    plt.close()

    scores = {'auroc': auroc, 'auprc': auprc, 'aurc': aurc}
    return scores


if __name__ == "__main__":
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="/app/data/llama2-7b-chat-hf")
    parser.add_argument("--amateur-model-name", type=str, default="/app/data/ep0.1")
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--output-path", type=str, default="./tfqa_result.json")
    parser.add_argument("--early_exit_layers", type=str, default="-1")
    parser.add_argument("--mode", type=str, default="contrastive-decoding")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--do-rating", action="store_true")
    parser.add_argument("--dataset_name", type=str, choices=["triviaqa", "natural_questions", "hotpotqa"],
                        default="natural_questions")
    parser.add_argument("--debug", type=bool, default=False)
    args = parser.parse_args()

    questions, answers = load_dataset_data(args.dataset_name, args.debug)

    llm = ContrastiveDecoding(args.model_name, device=args.device, max_gpu_memory=args.max_gpu_memory,
                              amateur_model_name=args.amateur_model_name)
    stop_words = ["Q:"]
    llm.set_stop_words(stop_words)

    generate_kwargs = dict(max_new_tokens=20, mode=args.mode)
    result_dict = {'qid_list': [], 'answers': {}, 'model_completion': {}, 'questions': {}, 'logit_scores': {}}

    for i, question in enumerate(tqdm(questions)):
        answer = answers[i]
        prompt = build_prompt(question, 'few_shot')
        model_completion, _ = llm.generate(prompt,** generate_kwargs)
        
        for stop_word in stop_words:
            length_to_remove = len(stop_word)
            if model_completion[-length_to_remove:] == stop_word:
                model_completion = model_completion[:-length_to_remove]
        
        if 'Q:' in model_completion:
            model_completion = model_completion.split('Q:')[0].strip()
        
        model_completion = model_completion.strip()

        result_dict['qid_list'].append(i)
        result_dict['answers'][i] = answer
        result_dict['model_completion'][i] = model_completion
        result_dict['questions'][i] = question

        if args.debug and i > 10:
            break

    if args.dataset_name in ['triviaqa', 'hotpotqa']:
        eval_metrics = evaluate_triviaqa(result_dict['answers'], result_dict['model_completion'])
    elif args.dataset_name == 'natural_questions':
        eval_metrics = evaluate_nq(result_dict['answers'], result_dict['model_completion'])
    else:
        raise NotImplementedError(f"Dataset {args.dataset_name} not implemented yet.")

    print(f"Exact Match（EM）: {eval_metrics['exact_match']:.2f}")
    print(f"F1: {eval_metrics['f1']:.2f}")
    
    os.makedirs(os.path.dirname(args.output_path), exist_ok=True)
    with open(args.output_path, 'w') as f:
        json.dump(result_dict, f)
    
    end = time.time()
    print(f"Total time: {end - start:.2f} seconds")