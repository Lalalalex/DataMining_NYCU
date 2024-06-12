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
import numpy as np
import random
import warnings
warnings.filterwarnings('ignore')

# os.environ['CUDA_VISIBLE_DEVICES']='3'
base_path = '/home/twhuang/homework/Data_Mining/HW2'

class Config:
    data_path = os.path.join(base_path, 'train.json')
    model_save_path = os.path.join(base_path, 'model_title/')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_length = 512
    batch_size = 128
    num_workers = 4
    learning_rate = 8e-4
    epoch = 11

def show_curve(data, file_name = 'learning_curve', title = 'Learning Curve', x = 'epoch', y = 'accuracy'):
    plt.figure()
    plt.title(title)
    for i in data:
        plt.plot(data[i], label = i)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.savefig(file_name)

def split_dataset(df, split = 0.95):
    df = df.sample(frac = 1).reset_index(drop = True)
    train_df = df.iloc[:int(len(df)*split)]
    valid_df = df.iloc[int(len(df)*split):]
    return train_df, valid_df

class Dataset(Dataset):
    def __init__(self, df, mode = 'train', title_max_length = 200, text_max_length = 512):
        self.mode = mode
        self.df = df
        self.tokenizer = AutoTokenizer.from_pretrained('textattack/bert-base-uncased-yelp-polarity')
        self.title_max_length = title_max_length
        self.text_max_length = text_max_length
    
    def __len__(self):
        return len(self.df)
    
    def random_deletion(self, text, p=0.5):
        words = text.split()
        if len(words) == 1:
            return text
        remaining = list(filter(lambda x: random.uniform(0, 1) > p, words))
        if len(remaining) == 0:
            return ' '.join([words[random.randint(0, len(words) - 1)]])
        else:
            return ' '.join(remaining)

    def __getitem__(self, index, augmentation = False):
        title = self.df.loc[index, 'title'] + ' ' + self.df.loc[index, 'text']
        text = self.df.loc[index, 'text']
        if augmentation:
            self.random_deletion(title)
            self.random_deletion(text)
        text_len = len(text)
        verified_purchase = self.df.loc[index, 'verified_purchase']
        helpful_vote = self.df.loc[index, 'helpful_vote']

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
                'helpful_vote' : helpful_vote
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


def train_epoch(model, dataloader, loss_function, optimizer):
    model.train()
    total_loss = 0
    total_f1_score = 0
    with tqdm(dataloader, unit = 'Batch', desc = 'Train') as tqdm_loader:
        for index, data in enumerate(tqdm_loader):
            label = torch.tensor(data['label'].to(device = cfg.device), dtype = torch.long)
            
            predict = model(input_ids = data['title_encoded_idf'].to(device = cfg.device),
                    token_type_ids = data['title_encoded_token'].to(device = cfg.device),
                    attention_mask = data['title_encoded_mask'].to(device = cfg.device)).logits
            
            loss = loss_function(predict, label)
            predict = predict.cpu().detach().argmax(dim = 1)
            label = label.cpu().detach()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss = loss.detach().item()
            total_loss = total_loss + loss
            total_f1_score = total_f1_score + f1_score(predict, label, average = 'macro', zero_division=0)
            
            tqdm_loader.set_postfix(loss = loss, average_loss = total_loss/(index + 1), average_f1_score = total_f1_score/(index + 1))

def valid_epoch(model, dataloader, loss_function, best_f1_score, best_loss):
    model.eval()
    total_loss = 0
    total_f1_score = 0
    average_loss = 0.0
    average_f1_score = 0.0
    with torch.no_grad():
        with tqdm(dataloader, unit = 'Batch', desc = 'Valid') as tqdm_loader:
            for index, data in enumerate(tqdm_loader):
                label = torch.tensor(data['label'].to(device = cfg.device), dtype = torch.long)
                
                predict = model(input_ids = data['title_encoded_idf'].to(device = cfg.device),
                        token_type_ids = data['title_encoded_token'].to(device = cfg.device),
                        attention_mask = data['title_encoded_mask'].to(device = cfg.device)).logits
                
                loss = loss_function(predict, label)
                predict = predict.cpu().detach().argmax(dim = 1)
                label = label.cpu().detach()

                loss = loss.detach().item()
                total_loss = total_loss + loss
                total_f1_score = total_f1_score + f1_score(predict, label, average = 'macro', zero_division=0)
                average_loss = total_loss/(index + 1)
                average_f1_score = total_f1_score/(index + 1)
                tqdm_loader.set_postfix(loss = loss, average_loss = average_loss, average_f1_score = average_f1_score, best_f1_score = best_f1_score)

        if average_f1_score > best_f1_score:
            print('best model update')
            best_f1_score = average_f1_score
            best_loss = average_loss
            torch.save(model, cfg.model_save_path + 'best_model')
        elif average_f1_score == best_f1_score:
            if average_loss <= best_loss:
                print('best model update')
                best_loss = average_loss
                torch.save(model, cfg.model_save_path + 'best_model')
    return best_f1_score, best_loss

cfg = Config()

data_json_file = open(cfg.data_path, 'r')
all_data = json.load(data_json_file)
all_df = pd.json_normalize(all_data)
# print(all_df.head(10))
train_df, valid_df = split_dataset(all_df)
train_dataset = Dataset(train_df.reset_index(drop=True))
train_dataloader = DataLoader(train_dataset, batch_size = cfg.batch_size, num_workers = cfg.num_workers, drop_last = False)

valid_dataset = Dataset(valid_df.reset_index(drop=True))
valid_dataloader = DataLoader(valid_dataset, batch_size = cfg.batch_size, num_workers = cfg.num_workers, drop_last = False)

model = BertForSequenceClassification.from_pretrained('textattack/bert-base-uncased-yelp-polarity')
model.classifier = nn.Linear(in_features = 768, out_features = 5, bias=True)
# print(model)
model = nn.DataParallel(model).to(cfg.device)
loss_function = nn.CrossEntropyLoss()
#optimizer = torch.optim.AdamW(model.parameters(), lr = cfg.learning_rate)
optimizer = Ranger21(model.parameters(), lr = cfg.learning_rate, num_epochs = cfg.epoch, num_batches_per_epoch = len(train_dataloader))

best_loss = 100
best_f1_score = 0

for epoch in range(cfg.epoch):
    print('Epoch ' + str(epoch + 1))
    train_epoch(model, train_dataloader, loss_function, optimizer)
    best_f1_score, best_loss = valid_epoch(model, valid_dataloader, loss_function, best_f1_score, best_loss)
    torch.save(model,cfg.model_save_path + 'model' + str(epoch + 1))