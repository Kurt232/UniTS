# evaluation classification based on gpt/bert embeddings
import os

import numpy as np
import math
import json
import string
import torch
import numpy as np
from collections import OrderedDict
from transformers import AutoTokenizer, BertModel
from sklearn.metrics import accuracy_score, classification_report
from util.stats import calculate_stats

dataset = 'fusion'
llm_task = 'cla'
text_embed_setting = 'bert'

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

base_path = '/data/wenhao/wjdu/sampling/results/UniTS_HEAD/'
eval_file_list = os.listdir(base_path)
eval_file_list = [ base_path + x for x in eval_file_list if x.endswith('.json')]

for x in eval_file_list:
    assert os.path.exists(x) == True

label_list = ["climbing stairs", "sitting", "standing", "walking", "descending stairs", "lying"]
num_class = len(label_list)
device = "cuda" if torch.cuda.is_available() else "cpu"
bert_mdl_size = 'bert-large-uncased'
bert_tokenizer = AutoTokenizer.from_pretrained(bert_mdl_size, model_max_length=512)
bert_model = BertModel.from_pretrained(bert_mdl_size).to(device)

all_acc = {}
for eval_file in eval_file_list:
    def get_bert_embedding(input_text):
        input_text = remove_punctuation_and_lowercase(input_text)
        #print(input_text)
        inputs = bert_tokenizer(input_text, return_tensors="pt")
        if inputs['input_ids'].shape[1] > 512:
            inputs['input_ids'] = inputs['input_ids'][:, :512]
            inputs['token_type_ids'] = inputs['token_type_ids'][:, :512]
            inputs['attention_mask'] = inputs['attention_mask'][:, :512]
            #print('trim the length')
            #print(inputs['input_ids'].shape)
        outputs = bert_model(**inputs.to(device))
        last_hidden_states = torch.mean(outputs.last_hidden_state[0], dim=0).cpu().detach().numpy()
        return last_hidden_states

    # def get_gpt_embedding(input_text, mdl_size='text-embedding-ada-002'):
    #     openai.api_key = 'your_open_ai_key'
    #     response = openai.Embedding.create(
    #         input=input_text,
    #         model=mdl_size
    #     )
    #     embeddings = response['data'][0]['embedding']
    #     return embeddings

    def cosine_similarity(vector1, vector2):
        dot_product = sum(v1 * v2 for v1, v2 in zip(vector1, vector2))
        magnitude1 = math.sqrt(sum(v1 ** 2 for v1 in vector1))
        magnitude2 = math.sqrt(sum(v2 ** 2 for v2 in vector2))
        return dot_product / (magnitude1 * magnitude2)

    def remove_punctuation_and_lowercase(text):
        text = text.translate(str.maketrans('', '', string.punctuation))
        text = text.lower()
        return text

    def gen_cm(all_truth, all_pred, save_name):
        from sklearn.metrics import confusion_matrix
        import matplotlib.pyplot as plt
        import numpy as np

        # list of label names
        label_names = list(label_dict.keys())

        # generate confusion matrix
        cm = confusion_matrix(all_truth, all_pred, labels=range(num_class))

        # plot confusion matrix as a figure
        plt.imshow(cm, cmap=plt.cm.Blues)
        plt.title("Confusion Matrix")
        plt.colorbar()
        tick_marks = np.arange(len(label_names))
        plt.xticks(tick_marks, label_names, rotation=90, fontsize=6)
        plt.yticks(tick_marks, label_names, fontsize=6)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # add label values to the confusion matrix cells
        for i in range(len(label_names)):
            for j in range(len(label_names)):
                val = 0
                if i < cm.shape[0] and j < cm.shape[1]:
                    val = cm[i, j]
                plt.text(j, i, val, ha="center", va="center", color="black")
        
        plt.savefig(save_name, dpi=300)
        plt.close()

    # load cached label embedding dict
    # if os.path.exists('../eval_res/label_embed_dict/{:s}_{:s}.json'.format(dataset, text_embed_setting)):
    if False:
        with open('../eval_res/label_embed_dict/{:s}_{:s}.json'.format(dataset, text_embed_setting), 'r') as f:
            json_str = f.read()
        label_dict = json.loads(json_str, object_pairs_hook=OrderedDict)
    else:
        label_dict = OrderedDict()
        for i in range(len(label_list)):
            class_code = i
            class_name = label_list[i]
            if text_embed_setting == 'bert':
                label_dict[class_name] = get_bert_embedding(class_name)
            # elif text_embed_setting == 'gpt':
            #     label_dict[class_name] = get_gpt_embedding(class_name)
        # with open('../eval_res/label_embed_dict/{:s}_{:s}.json'.format(dataset, text_embed_setting), 'w') as f:
        #     print(label_dict)
        #     json_str = json.dumps(label_dict)
        #     f.write(json_str)

    with open(eval_file, 'r') as fp:
        eval_data = json.load(fp)

    if os.path.exists('../eval_res/embedding_cache/{:s}_{:s}_{:s}.json'.format( dataset, llm_task, text_embed_setting)) == True:
        with open('../eval_res/embedding_cache/{:s}_{:s}_{:s}.json'.format( dataset, llm_task, text_embed_setting), 'r') as f:
            embed_cache = f.read()
        embed_cache = json.loads(embed_cache)
    else:
        embed_cache = {}

    def get_pred(cur_pred_list, label_dict, mode='max'):
        ["climbing stairs", "sitting", "standing", "walking", "descending stairs", "lying"]
        if 'climb' in cur_pred_list.lower():
            return 0
        elif 'sit' in cur_pred_list.lower() or 'itting' in cur_pred_list.lower():
            return 1
        elif 'stand' in cur_pred_list.lower():
            return 2
        elif 'walk' in cur_pred_list.lower():
            return 3
        elif 'descend' in cur_pred_list.lower():
            return 4
        elif 'ly' in cur_pred_list.lower() or 'lie' in cur_pred_list.lower():
            return 5
            
        # at beginning, all zero scores
        score = np.zeros(num_class)
        label_embed_list = list(label_dict.values())
        # pred might not be a single text
        for cur_pred in cur_pred_list:
            if cur_pred in embed_cache:
                cur_pred_embed = embed_cache[cur_pred]
            else:
                # if text_embed_setting == 'gpt':
                #     cur_pred_embed = get_gpt_embedding(cur_pred)
                # else:
                cur_pred_embed = get_bert_embedding(cur_pred)
                embed_cache[cur_pred] = cur_pred_embed
            for i in range(num_class):
                if mode == 'accu':
                    score[i] = score[i] + cosine_similarity(cur_pred_embed, label_embed_list[i])
                elif mode == 'max':
                    score[i] = max(score[i], cosine_similarity(cur_pred_embed, label_embed_list[i]))
        cur_pred = np.argmax(score)
        return cur_pred

    num_sample = len(eval_data)
    print('number of samples {:d}'.format(num_sample))
    all_pred = np.zeros([num_sample, num_class])
    all_truth = np.zeros([num_sample, num_class])
    for i in range(num_sample):
        cur_id = eval_data[i]['data_id']
        if llm_task == 'cla':
            cur_pred_list = eval_data[i]['pred']
        elif llm_task == 'caption':
            cur_pred_list = eval_data[i]['pred']
        
        cur_pred_list : str = cur_pred_list
        # cur_pred_list = cur_pred_list.split('### Response:')[-1]
        # cur_pred_list = cur_pred_list.strip()

        if len(cur_pred_list) >=20:
            cur_pred_list = cur_pred_list[len(cur_pred_list)-20: ]
        
        cur_truth = eval_data[i]['ref'].split('\n')[0].strip()
        if cur_truth[1:].startswith('.'):
            cur_truth = cur_truth[3:]
        elif cur_truth.startswith('Answer: '):
            cur_truth = cur_truth[11:]
        
        cur_truth_idx = list(label_dict.keys()).index(cur_truth)
        print(cur_truth_idx)
        cur_pred_idx = get_pred(cur_pred_list, label_dict)
        print('Truth: ', cur_truth_idx, list(label_dict.keys())[cur_truth_idx], 'Pred: ', cur_pred_idx, list(label_dict.keys())[cur_pred_idx], cur_truth_idx==cur_pred_idx, cur_pred_list)
        all_pred[i, cur_pred_idx] = 1.0
        all_truth[i, cur_truth_idx] = 1.0
    
    save_fold = os.path.join(base_path, "{:s}_cla_report".format('.'.join(eval_file.split('/')[-1].split('.')[:-1])))
    if os.path.exists(save_fold) == False:
        os.makedirs(save_fold)

    # np.save(save_fold + '/all_pred.npy', all_pred)
    # np.save(save_fold + '/all_truth.npy', all_truth)
    stats = calculate_stats(all_pred, all_truth)

    mAP = np.mean([stat['AP'] for stat in stats])
    mAUC = np.mean([stat['auc'] for stat in stats])
    acc = stats[0]['acc']

    np.savetxt(save_fold + '/result_summary.csv', [mAP, mAUC, acc], delimiter=',')

    # embed_cache = json.dumps(embed_cache)
    # save_cache_path = '../eval_res/embedding_cache/{:s}_{:s}_{:s}.json'.format(dataset, llm_task, text_embed_setting)
    # with open(save_cache_path, 'w') as f:
    #     f.write(embed_cache)

    sk_acc = accuracy_score(all_truth, all_pred)
    print('accuracy: ', acc, sk_acc)

    report = classification_report(all_truth, all_pred, target_names=list(label_dict.keys()))
    with open(save_fold + "/cla_summary.txt", "w") as f:
        f.write(report)
    gen_cm(all_truth.argmax(axis=1), all_pred.argmax(axis=1), save_fold + "/cm.png")

    all_acc[eval_file.split("/")[-1]] = acc

print(json.dumps(all_acc, indent=2, sort_keys=True))
