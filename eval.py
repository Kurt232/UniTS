import os
import numpy as np
import json
import torch
from collections import OrderedDict
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

# os.environ["CUDA_VISIBLE_DEVICES"] = "3"

base_path = '/data/wjdu/multi/ds'
eval_file_list = os.listdir(base_path)
eval_file_list = [ os.path.join(base_path, x) for x in eval_file_list if x.endswith('.json') and x.startswith('ViT_LLM')]

for x in eval_file_list:
    assert os.path.exists(x) == True

# Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
label_list = ['downstairs', 'jog', 'lie', 'sit', 'stand', 'upstairs', 'walk']
bert_model = BertModel.from_pretrained('bert-large-uncased').to(device)
bert_tokenizer = AutoTokenizer.from_pretrained('bert-large-uncased', model_max_length=512)

# Helper functions
def get_bert_embedding(text):
    inputs = bert_tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
    outputs = bert_model(**inputs.to(device))
    return torch.mean(outputs.last_hidden_state[0], dim=0).cpu().detach().numpy()

def cosine_similarity(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

def classify_text(pred_text, label_dict):
    pred_text = pred_text.lower()
    if 'down' in pred_text:
        return 0, False
    elif 'jog' in pred_text:
        return 1, False
    elif 'lie' in pred_text:
        return 2, False
    elif 'sit' in pred_text:
        return 3, False
    elif 'stand' in pred_text:
        return 4, False
    elif 'up' in pred_text:
        return 5, False
    elif 'walk' in pred_text:
        return 6, False
    else:
        scores = [cosine_similarity(get_bert_embedding(pred_text), embed) for embed in label_dict.values()]
        return np.argmax(scores), True

def preprocess_text(text):
    if len(text) >= 20:
        text = text[-20:]
    return text

def preprocess_truth(truth):
    truth = truth.split('\n')[0].strip()
    if truth.startswith('.') and truth[1:].startswith('.'):
        truth = truth[3:]
    elif truth.startswith('Answer: '):
        truth = truth[11:]
    return truth

def plot_confusion_matrix(cm, labels, save_path):
    plt.imshow(cm, cmap="Blues")
    plt.colorbar()
    plt.xticks(np.arange(len(labels)), labels, rotation=90)
    plt.yticks(np.arange(len(labels)), labels)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    
    # Add label values to the confusion matrix cells
    for i in range(len(labels)):
        for j in range(len(labels)):
            plt.text(j, i, cm[i, j], ha="center", va="center", color="black")
    
    plt.savefig(save_path, dpi=300)
    plt.close()

# Load and prepare label embeddings
label_dict = OrderedDict((label, get_bert_embedding(label)) for label in label_list)
acc_dict = {}

# Evaluation
for eval_file in eval_file_list:
    out_of_list_embeddings = []
    error_list = []
    with open(eval_file, 'r') as f:
        data = json.load(f)

    num_sample = len(data)
    num_class = len(label_list)
    all_pred = np.zeros(num_sample, dtype=int)
    all_truth = np.zeros(num_sample, dtype=int)
    
    for idx, item in enumerate(data):
        truth_text = preprocess_truth(item['ref'])
        # pred_text = preprocess_text(item['pred'])
        pred_text = item['pred']
        truth_idx = label_list.index(truth_text)
        pred_idx, is_out = classify_text(pred_text, label_dict)
        if is_out:
            item['pred_label'] = label_list[pred_idx]
            out_of_list_embeddings.append(item)
        all_truth[idx] = truth_idx
        all_pred[idx] = pred_idx
        if truth_idx != pred_idx:
            error_list.append(item)

    # Metrics and saving results
    acc = accuracy_score(all_truth, all_pred)
    report = classification_report(all_truth, all_pred, target_names=label_list, labels=range(num_class), output_dict=True)
    report1 = classification_report(all_truth, all_pred, target_names=label_list, labels=range(num_class))
    cm = confusion_matrix(all_truth, all_pred, labels=range(num_class))
    
    save_base = os.path.splitext(eval_file)[0]
    if not os.path.exists(save_base):
        os.makedirs(save_base)
    with open(f"{save_base}/report.txt", "w") as f:
        f.write(json.dumps(report, indent=2))
    with open(f"{save_base}/report1.txt", "w") as f:
        f.write(report1)
    plot_confusion_matrix(cm, label_list, f"{save_base}/cm.png")
    acc_dict[os.path.basename(eval_file).split('.')[0]] = acc

    # Save out-of-list embeddings for review
    if len(out_of_list_embeddings) > 0:
        with open(f"{save_base}/out_of_list.json", "w") as f:
            json.dump(out_of_list_embeddings, f, indent=2)
    if len(error_list) > 0:
        with open(f"{save_base}/error_list.json", "w") as f:
            json.dump(error_list, f, indent=2)

print(json.dumps(acc_dict, indent=4, sort_keys=True))