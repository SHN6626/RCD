# Ref: https://github.com/kojima-takeshi188/zero_shot_cot

import re
import os
import json
import random
import torch
import numpy as np
import pandas as pd
import transformers
from tqdm import tqdm, trange
import argparse

import ssl
import urllib.request
import zipfile
import sys
from pathlib import Path 
file = Path(__file__).resolve()
parent, root = file.parent, file.parents[1]
sys.path.append(str(root))

transformers.logging.set_verbosity(40)
from decoding_algorithm import ContrastiveDecoding

ANS_RE = re.compile(r"#### (\-?[0-9\.\,]+)")
INVALID_ANS = "[invalid]"

N_SHOT = 8
COT_FLAG = True
DEBUG = True
ANSWER_TRIGGER = "The answer is"

LLAMA2_Truth_PROMPT = {
    "description": "Llama 2 chat one shot prompt",
    "prompt": '''[INST] <<SYS>>
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe.  Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

You need to extract the word or phrase that represents the fact of the sentence from the following sentence (more likely to be a noun):
<</SYS>>

{instruction} [/INST] '''
}

def load_csv(file_path):
    # Format of each line:
    # {'instruction': ..., 'input': ..., 'output':...}
    '''
    Data format:

    ,full_prefix,doc_id,completion,contradiction_0,contradiction_1,contradiction_2,longest_completions,turncated_prefixes
    0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. ",0,Whether or not it gets a second season of The Witcher is another question.,Whether or not it gets a second season of Stranger Things is another question.,Whether or not it gets a fifth season of The Witcher is another question.,Whether or not it gets a second season of Black Mirror is another question.,15.0,"As streaming television services continue to gain market share, there are a number of reasons why Netflix might be in trouble. Time Warner is taking its HBO content online, Amazon offers premium content for a monthly fee, and Hulu has reached nine million users. While these competitors may cause a bit of worry, it’s not the end of the world. Although Netflix has a huge amount of potential, the increased competition is unlikely to hurt its profitability.
    While the global pandemic last year caused a major shakeup in Hollywood, Netflix should not rest on its laurels. With a variety of rivals on the rise, it’s unlikely that it can continue to rely on its current performance. Despite the competition, the company has made a number of impactful moves across the board, including clamping down on password sharing. And in the coming years, Netflix should continue to grow and compete with new competitors.
    With more competitors entering the streaming space, Netflix is likely to face a more difficult time keeping its current market share. Disney has been investing heavily in the service and Amazon is expected to do the same. Both companies expect to add 35-40 million subscribers per year through 2024. Despite the competition, Netflix still remains the top streaming service. Its lack of original content has hurt its numbers in the last few quarters. Its only big original hit in the US was Cobra Kai, which only got four seasons. "

    '''
    list_data_dict = []
    df = pd.read_csv(file_path)
    if 'news' in file_path:
        prefix_type = 'full_prefix'
    else:
        prefix_type = 'turncated_prefixes'
    for idx in range(len(df)):
        item = dict(
            prefix=df[prefix_type][idx],
            completion=df['completion'][idx],
            contradiction_0=df['contradiction_0'][idx],
            contradiction_1=df['contradiction_1'][idx],
            contradiction_2=df['contradiction_2'][idx],
        )
        list_data_dict.append(item)
    return list_data_dict

def download_url(url: str, folder='folder'):
    """
    Downloads the content of an url to a folder. Modified from \
    https://github.com/pyg-team/pytorch_geometric/tree/master/torch_geometric

    Args:
        url (string): The url of target file.
        folder (string): The target folder.

    Returns:
        string: File path of downloaded files.
    """

    file = url.rpartition('/')[2]
    file = file if file[0] == '?' else file.split('?')[0]
    path = os.path.join(folder, file)
    if os.path.exists(path):
        print(f'File {file} exists, use existing file.')
        return path

    print(f'Downloading {url}')
    os.makedirs(folder, exist_ok=True)
    ctx = ssl._create_unverified_context()
    data = urllib.request.urlopen(url, context=ctx)
    with open(path, 'wb') as f:
        f.write(data.read())

    return path

def extract_answer_from_output(completion):
    match = ANS_RE.search(completion)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    else:
        return INVALID_ANS


def is_correct(model_answer, answer):
    gt_answer = extract_answer_from_output(answer)
    assert gt_answer != INVALID_ANS
    return model_answer == gt_answer


def clean_answer(model_pred):
    model_pred = model_pred.lower()
    preds = model_pred.split(ANSWER_TRIGGER.lower())
    answer_flag = True if len(preds) > 1 else False
    if answer_flag:
        # Pick first answer with flag
        pred = preds[1]
    else:
        # Pick last number without flag
        pred = preds[-1]

    pred = pred.replace(",", "")
    pred = [s for s in re.findall(r'-?\d+\.?\d*', pred)]

    if len(pred) == 0:
        return INVALID_ANS

    if answer_flag:
        # choose the first element in list
        pred = pred[0]
    else:
        # choose the last element in list
        pred = pred[-1]

    # (For arithmetic tasks) if a word ends with period, it will be omitted ...
    if pred[-1] == ".":
        pred = pred[:-1]

    return pred

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, default="huggyllama/llama-7b")
    parser.add_argument("--amateur-model-name", type=str, default=None)
    parser.add_argument("--num-gpus", type=str, default="1")
    parser.add_argument("--amateur-model-nums-gpus", type=str, default="1") 
    parser.add_argument("--max_gpu_memory", type=int, default=27)
    parser.add_argument("--device", type=str, choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--is-chat", action="store_true")
    parser.add_argument("--output-path", type=str, default="./gsm8k_result")
    parser.add_argument("--data-path", type=str, default="./gsm8k_result")
    # parallel mode (split the dataset into multiple parts, inference by separate processes)
    parser.add_argument("--early-exit-layers", type=str, default="-1")
    parser.add_argument("--parallel", action="store_true")
    parser.add_argument("--total-shard", type=int, default=8)
    parser.add_argument("--mode", type=str, choices=["greedy", "contrastive-decoding", "dola", "prompt-contrastive-decoding","dual-contrastive-decoding","multi-contrastive-decoding"], default="greedy")
    parser.add_argument("--shard-id", type=int, default=None)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--top_k", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--relative_top", type=float, default=0.0)
    parser.add_argument("--relative_top_value", type=float, default=-1000.0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--do_shuffle", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retry", type=int, default=1)
    parser.add_argument("--lam", type=float, default=1)
    parser.add_argument("--savepath", type=str, default="logits/")
    args = parser.parse_args()
    model_name = args.model_name
    num_gpus = args.num_gpus
    device = args.device

    # Get test file
    fp = os.path.join(args.data_path)
    if not os.path.exists(fp):
        raise ValueError(f"Test file {fp} does not exist.")
    print(fp)
    list_data_dict = load_csv(fp)

    if args.parallel:
        chunk_size = len(list_data_dict) // args.total_shard
        list_data_dict = list_data_dict[args.shard_id * chunk_size: (args.shard_id + 1) * chunk_size]

    if args.debug:
        list_data_dict = list_data_dict[:10]
    
    #llm = DoLa(model_name, device, num_gpus, args.max_gpu_memory)
    llm = ContrastiveDecoding(model_name, device, args.max_gpu_memory, args.amateur_model_name, num_gpus=int(args.num_gpus), amateur_model_nums_gpus=int(args.amateur_model_nums_gpus))
    llm.set_stop_words(["Q:", "\end{code}"])
    early_exit_layers = [int(x) for x in args.early_exit_layers.split(',')]

    if args.mode == "contrastive-decoding":
        assert args.amateur_model_name is not None
        print("MODE: constrastive decoding between model1: {:s} and model2: {:s}".format(args.model_name, args.amateur_model_name), flush=True)
        mode = "contrastive-decoding"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    elif args.mode == "dola":
        if len(early_exit_layers) == 2:
            print(f"MODE: DoLa-static decoding with mature layer: {early_exit_layers[1]} and premature layer: {early_exit_layers[0]}")
            mode = "dola-static"
            mature_layer = early_exit_layers[1]
            premature_layer = early_exit_layers[0]
            candidate_premature_layers = None
        else:
            print(f"MODE: DoLa decoding with mature layer: {early_exit_layers[-1]} and premature layers: {early_exit_layers[:-1]}")
            mode = "dola"
            mature_layer = early_exit_layers[-1]
            premature_layer = None
            candidate_premature_layers = early_exit_layers[:-1]
            premature_layer_dist = {l:0 for l in candidate_premature_layers}
    elif args.mode == "greedy":
        print("MODE: naive (greedy) decoding from the last layer", flush=True)
        mode = "baseline"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    elif args.mode == "prompt-contrastive-decoding":
        print("MODE: constrastive decoding with evil prompt", flush=True)
        mode = "prompt-contrastive-decoding"
        mature_layer = None
        premature_layer = None
        candidate_premature_layers = None
    # elif args.mode == "dual-contrastive-decoding":
    #     assert args.amateur_model_name is not None
    #     print("MODE: dual-constrastive decoding between model1: {:s} and model2: {:s}".format(args.model_name, args.amateur_model_name), flush=True)
    #     mode = "dual-contrastive-decoding"
    #     mature_layer = None
    #     premature_layer = None
    #     candidate_premature_layers = None
    # elif args.mode == "multi-contrastive-decoding":
    #     assert args.amateur_model_name is not None
    #     print("MODE: multi-constrastive decoding between model1: {:s} and model2: {:s}".format(args.model_name, args.amateur_model_name), flush=True)
    #     mode = "multi-contrastive-decoding"
    #     mature_layer = None
    #     premature_layer = None
    #     candidate_premature_layers = None
    else:
        raise NotImplementedError

    answers = []
    result_dict = {'is_correct': [], 'model_answer': [], 'model_completion': [], 'full_input_text': [], 'result': []}

    for sample in tqdm(list_data_dict):
        context = sample['prefix']
        answer_true = ' ' + sample['completion']
        answers_false = []
        truth_prompt = LLAMA2_Truth_PROMPT["prompt"]
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        generate_kwargs = dict(max_new_tokens=args.max_new_tokens, repetition_penalty=args.repetition_penalty, mode=mode, mature_layer=mature_layer, premature_layer=premature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value, post_softmax=False, lam=args.lam,savepath=args.savepath)
        answer_true_log_prob, c_dist = llm.lm_score(context, answer_true, truth_prompt=truth_prompt, **generate_kwargs)
        
        if mode == "dola":
            for k, v in c_dist.items():
                premature_layer_dist[k] += v
        answer_false_log_probs = []
        for answer_false in answers_false:
            answer_false_log_prob, c_dist = llm.lm_score(context, answer_false, truth_prompt=truth_prompt, **generate_kwargs)
            if mode == "dola":
                for k, v in c_dist.items():
                    premature_layer_dist[k] += v
            answer_false_log_probs.append(answer_false_log_prob)
        if args.debug:
            print(f'log prob of answers: {answer_true_log_prob}', end=' ')
            for answer_false_log_prob in answer_false_log_probs:
                print(f'{answer_false_log_prob}', end=' ')
            print()
        is_cor = True
        for answer_false_log_prob in answer_false_log_probs:
            if answer_true_log_prob < answer_false_log_prob:
                is_cor = False
                break
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)
        result_dict['model_completion'].append([answer_true_log_prob] + answer_false_log_probs)

        print(f'Num of total question: {len(answers)}, '
            f'correct num: {sum(answers)}, '
            f'correct rate: {float(sum(answers))/len(answers)}.')
    correct_rate = float(sum(answers)/len(answers))
    result_dict['result'].append(correct_rate)
    if mode == "dola" and args.debug:
        total_tokens = sum(premature_layer_dist.values())
        if total_tokens > 0:
            for l in candidate_premature_layers:
                print('Premature layer {0} was used {1} times, {2}%'.format(l, premature_layer_dist[l], round(premature_layer_dist[l] / total_tokens * 100, 2)))
    # save results to a json file
    model_tag = model_name.split('/')[-1] if model_name[-1] != '/' else model_name.split('/')[-2]
    output_file = args.output_path if args.shard_id is None else (args.output_path+"_"+str(args.shard_id)+".json")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    with open(output_file, 'w') as f:
        json.dump(result_dict, f)
    
    print(f"{float(sum(answers))/len(answers)}")