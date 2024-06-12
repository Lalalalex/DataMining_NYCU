import json
import os
import pandas as pd
from transformers import AutoTokenizer, BertForSequenceClassification
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from sklearn.metrics import f1_score
from ranger21 import Ranger21
from tqdm import tqdm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np

# os.environ['CUDA_VISIBLE_DEVICES']='3'
base_path = '/home/twhuang/homework/Data_Mining/HW2'

class Config:
    data_path = os.path.join(base_path, 'test.json')
    model_save_path = os.path.join(base_path, 'model_title/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 512
    batch_size = 128
    num_workers = 4
    learning_rate = 3e-3
    epoch = 50

def split_dataset(df, split = 0.9):
    df = df.sample(frac = 1).reset_index(drop = True)
    train_df = df.iloc[:int(len(df)*split)]
    valid_df = df.iloc[int(len(df)*split):]
    return train_df, valid_df

def show_curve(data, file_name = 'learning_curve', title = 'Learning Curve', x = 'epoch', y = 'accuracy'):
    plt.figure()
    plt.title(title)
    for i in data:
        plt.plot(data[i], label = i)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.savefig(file_name)

class Dataset(Dataset):
    def __init__(self, df, mode = 'train', title_max_length = 128, text_max_length = 512):
        self.mode = mode
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-yelp-polarity')
        self.title_max_length = title_max_length
        self.text_max_length = text_max_length
    
    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        title = self.df.loc[index, 'title'] + ' ' + self.df.loc[index, 'text']
        text = self.df.loc[index, 'text']
        text_len = len(text)
        verified_purchase = self.df.loc[index, 'verified_purchase']
        helpful_vote = self.df.loc[index, 'helpful_vote']
        index = self.df.loc[index, 'index']

        title_encoded = self.tokenizer.encode_plus(
            title,
            add_special_tokens = True,
            max_length = self.title_max_length,
            padding = 'max_length',
            return_attention_mask = True,
            truncation = True
        )
        text_encoded = self.tokenizer.encode_plus(
            text,
            add_special_tokens = True,
            max_length = self.text_max_length,
            padding = 'max_length',
            return_attention_mask = True,
            truncation = True
        )

        if self.mode == 'test':
            return {
                'title_encoded_idf' : torch.tensor(title_encoded['input_ids']),
                'title_encoded_token' : torch.tensor(title_encoded['token_type_ids']),
                'title_encoded_mask' : torch.tensor(title_encoded['attention_mask']),
                'text_encoded_idf' : torch.tensor(text_encoded['input_ids']),
                'text_encoded_token' : torch.tensor(text_encoded['token_type_ids']),
                'text_encoded_mask' : torch.tensor(text_encoded['attention_mask']),
                'verified_purchase' : verified_purchase,
                'helpful_vote' : helpful_vote,
                'index': str('index_' + str(index)),
                'text_len': text_len
            }
        
        label = int(self.df.loc[index, 'rating']) - 1
        return {
                'title_encoded_idf' : torch.tensor(title_encoded['input_ids']),
                'title_encoded_token' : torch.tensor(title_encoded['token_type_ids']),
                'title_encoded_mask' : torch.tensor(title_encoded['attention_mask']),
                'text_encoded_idf' : torch.tensor(text_encoded['input_ids']),
                'text_encoded_token' : torch.tensor(text_encoded['token_type_ids']),
                'text_encoded_mask' : torch.tensor(text_encoded['attention_mask']),
                'verified_purchase' : verified_purchase,
                'helpful_vote' : helpful_vote,
                'label': label,
                'text_len': text_len
            }


def test(model, dataloader):
    with torch.no_grad():
        model.eval()
        predict_list = []
        predict_id_list = []
        text_len_list = []
        with tqdm(dataloader, unit = 'Batch', desc = 'Test') as tqdm_loader:
            for index, data in enumerate(tqdm_loader):
                predict = model(input_ids = data['title_encoded_idf'].to(device = cfg.device),
                        token_type_ids = data['title_encoded_token'].to(device = cfg.device),
                        attention_mask = data['title_encoded_mask'].to(device = cfg.device)).logits
                
                predict = predict.cpu().detach().argmax(dim = 1) + 1
                predict_list.append(predict)
                predict_id_list.append(data['index'])
                text_len_list.append(data['text_len'])
        predict_list = np.concatenate(predict_list, axis=0)
        predict_id_list =  np.concatenate(predict_id_list, axis=0)
        # text_len_list = list(np.concatenate(text_len_list, axis=0))
        # unique_values, counts = np.unique(text_len_list, return_counts=True)

        # plt.bar(counts, unique_values)
        # plt.savefig('length')
        # plt.xlabel('conuts')
        # plt.ylabel('length')
        # print(text_len_list)
    return predict_id_list, predict_list, text_len_list

cfg = Config()


data_json_file = open(cfg.data_path, 'r')
test_data = json.load(data_json_file)
test_df = pd.json_normalize(test_data)
test_df['index'] = range(0, len(test_df))
test_dataset = Dataset(test_df, mode = 'test')
test_dataloader = DataLoader(test_dataset, batch_size = cfg.batch_size, num_workers = cfg.num_workers, drop_last = False)

model = torch.load(cfg.model_save_path + 'best_model')
# print(model)
model.to(cfg.device)

id_list, label_list, text_len_list = test(model, test_dataloader)
# show_curve({'length': text_len_list})

submit_df  = pd.DataFrame()
submit_df['index'] = id_list
submit_df['rating'] = label_list
submit_df.to_csv(os.path.join('./', 'submission.csv'),index = False)