import argparse
import json

import evaluate
import torch

import mauve

from sklearn.metrics import f1_score
import numpy as np
from nltk import sent_tokenize

torch.manual_seed(0)
from transformers import set_seed
set_seed(0)


from summac.model_summac import SummaCZS

rouge = evaluate.load('rouge')

def load_jsonl(filename):
    d_list = []
    with open(filename, encoding='utf-8', mode='r') as in_f:
        print("Load Jsonl:", filename)
        for line in in_f:
            item = json.loads(line.strip())
            d_list.append(item)

    return d_list

def compute_f1(prediction, ground_truths):
    f1_sc = f1_score(ground_truths, prediction, average='macro')
    return {"F1": f1_sc}

def compute_mauve(prediction, ground_truths):

    human_data = []
    model_data = []
    for p,g in zip(prediction, ground_truths):
        human_data.append(g)
        model_data.append(p)

    out = mauve.compute_mauve(
        p_text=human_data,
        q_text=model_data,
        device_id=0,
        mauve_scaling_factor=3,
        max_text_length=256,
        verbose=False,
        batch_size=8,
        featurize_model_name="gpt2-large"
    )
    return out.mauve*0.01

def rouge_score(prediction, ground_truths):
    results = rouge.compute(predictions=prediction,references=ground_truths)
    return results

def lcs(s1, s2): 
   
    m = [[0 for i in range(len(s2)+1)] for j in range(len(s1)+1)] 
    mmax = 0   
    p = 0  
    for i in range(len(s1)):
        for j in range(len(s2)):
            if s1[i] == s2[j]:
                m[i+1][j+1] = m[i][j] + 1
                if m[i+1][j+1] > mmax:
                    mmax = m[i+1][j+1]
                    p = i+1
    return s1[p-mmax:p], mmax   
 
def get_unique_sentences(document):
    unique_sentences = []
    sentences = sent_tokenize(document)
    for sentence in sentences:
        if len(unique_sentences) == 0:
            unique_sentences.append(sentence)
            continue
        o_len = len(unique_sentences)
        find_one = False
        for ui in range(o_len):
            unis = unique_sentences[ui]
            X = sentence if len(sentence) <= len(unis) else unis
            Y = sentence if len(sentence) > len(unis) else unis

            lcs1,mlcs =  lcs(X.lower(), Y.lower())
            if mlcs >= 0.5*len(X):
                unique_sentences[ui] = Y
                find_one = True
                break
        if not find_one:
            unique_sentences.append(sentence)

    return ' '.join(unique_sentences)

def compute_summacc(prediction, ground_truths):
    model_zs = SummaCZS(granularity="paragraph-sentence", model_name="vitc", device="cuda",use_con=False)
    results = {"summacc":[]}
    
    for g, p in zip(ground_truths,prediction):
        p1 = get_unique_sentences(p)
        
        o1 = model_zs.score([g], [p1])
        o2 = model_zs.score([p1], [g])
        a = o1['images']
        b = o2['images']
        all_pairs = a[0].shape[-1]+b[0].shape[-1]
        wa = 1.0*a[0].shape[-1]/all_pairs
        wb = 1.0*b[0].shape[-1]/all_pairs
        score_zs2 = o1['scores']
        score_zs2_r = o2['scores']
        tmp = wa*score_zs2[0]+wb*score_zs2_r[0]

        results["summacc"] += [tmp]

    return np.mean(results["summacc"])

def compute_metric(test_file, output_file, metric_type, label_file=None):

    ground_truth = {}
    test_data = load_jsonl(test_file)

    if metric_type == "classification":
        labels = json.load(open(label_file))
    for d in test_data:
        if metric_type == "classification":
            label = labels[str(d["id"])]
            ground_truth[d["id"]] = label
        else:
            ground_truth[d["id"]] = d["target"]

    all_data = {}

    gts, preds = [], []

    data_records = load_jsonl(f"{output_file}")
    for d in data_records:
        if metric_type == "classification":
            preds.append(d["cls_pred"])
        else:
            preds.append(d["generation"])
        gts.append(ground_truth[d["id"]])

    print(f"Eval num instances: {len(gts)}")
    if metric_type == "classification":
        results = compute_f1(preds, gts)
        all_data = {metric:value*100 for metric,value in results.items()}
    else:
        results = rouge_score(preds, gts)
        all_data = {metric:value*100 for metric,value in results.items()}

        results = compute_mauve(preds, gts)*100.0
        results={"mauve":results}
        for metric,value in results.items():
            all_data[metric]= value*100
        
        results = compute_summacc(preds, gts)
        results={"summacc":results}
        for metric,value in results.items():
            all_data[metric]= value*100

    for metric in all_data.keys():
        v = all_data[metric]
        print(f"{metric}: {v}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-test_file", default=None)
    parser.add_argument("-output_file", default=None)
    parser.add_argument("-metric_type", default="generation")
    parser.add_argument("-label_file", default=None)
    
    args = parser.parse_args()

    compute_metric(args.test_file, args.output_file, args.metric_type, args.label_file)