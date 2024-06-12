from config import Config
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import numpy as np

class Dataset(Dataset):
    def __init__(self, behaviors_path, news_path, embeddings_parh, is_test_model = False):
        self.behaviors_df = pd.read_csv(behaviors_path, sep='\t')
        self.news_df = pd.read_csv(news_path, sep='\t')
        self.embeddings_df = pd.read_csv(embeddings_parh)
        self.is_test_model = is_test_model
        self.categoriacl_cols = ['category', 'subcategory']
        self.news_df = self.label_encode(self.news_df)

    def __len__(self):
        return len(self.behaviors_df)
    
    def get_news_info(self, news_id):
        news_info = self.news_df[self.news_df['news_id'] == news_id].iloc[0]
        category = news_info['category']
        sub_category = news_info['subcategory']

        return {
            'category': category, 
            'subcategory': sub_category
        }

    def label_encode(self, df):
        for ft in self.categoriacl_cols:
            le = pd.get_dummies
            le = LabelEncoder()
            le.fit(df[ft])
            df[ft]=le.transform(df[ft])
        return df
    
    def __getitem__(self, index):
        id = int(self.behaviors_df['id'][index])
        clicked_news = self.behaviors_df['clicked_news'][index].split()
        clicked_news_num = len(clicked_news)
        clicked_news_infos = []
        for news in clicked_news:
            clicked_news_infos.append(self.get_news_info(news))

        if self.is_test_model:
            impression_news = self.behaviors_df['impressions'][index].split()
            impression_news_infos = []
            for news in impression_news:
                impression_news_infos.append(self.get_news_info(news))
            
            return {
                'id': id,
                'clicked_news': clicked_news,
                'clicked_news_infos': clicked_news_infos,
                'impression_news': impression_news,
                'impression_news_infos': impression_news_infos,
                'clicked_news_num': clicked_news_num
            }

        impressions_all = self.behaviors_df['impressions'][index].split()
        impression_news = [impressions_data.split('-')[0] for impressions_data in impressions_all]
        impression_news_infos = []
        for news in impression_news:
            impression_news_infos.append(self.get_news_info(news))
        impression_labels = [int(impressions_data.split('-')[1]) for impressions_data in impressions_all]

        return {
            'id': id,
            'clicked_news': clicked_news,
            'clicked_news_infos': clicked_news_infos,
            'impression_news': impression_news,
            'impression_news_infos': impression_news_infos,
            'impression_labels': impression_labels,
            'clicked_news_num': clicked_news_num
        }

cfg = Config()
train_dataset = Dataset(cfg.train_behaviors_path, cfg.train_news_path, cfg.train_embeddings_path)
test_dataset = Dataset(cfg.test_behaviors_path, cfg.test_news_path, cfg.test_embeddings_path, is_test_model = True)

def train(dataset):
    total_accuracy = 0
    with tqdm(dataset, unit = 'data', desc = 'Train') as tqdm_loader:
        for index, data in enumerate(tqdm_loader):
            clicked_news_infos = data['clicked_news_infos']
            impression_news_infos = data['impression_news_infos']
            label = data['impression_labels']

            predict = [1 for i in range(len(impression_news_infos))]

            for i, impression_news in enumerate(impression_news_infos):
                for clicked_news in clicked_news_infos:
                    if clicked_news['category'] == impression_news['category']:
                        break
                    predict[i] = 0
            
            accuracy = accuracy_score(predict, label)
            total_accuracy = total_accuracy + accuracy
            average_accuracy = total_accuracy/(index + 1)
            tqdm_loader.set_postfix(average_accuracy = average_accuracy, accuracy = accuracy)

def test(dataset, sample_df):
    total_accuracy = 0
    predict_list = []
    with tqdm(dataset, unit = 'data', desc = 'Test') as tqdm_loader:
        for index, data in enumerate(tqdm_loader):
            id = data['id']
            clicked_news_infos = data['clicked_news_infos']
            impression_news_infos = data['impression_news_infos']

            predict = [id] + [1] * (len(impression_news_infos))

            for i, impression_news in enumerate(impression_news_infos):
                for clicked_news in clicked_news_infos:
                    if clicked_news['category'] == impression_news['category']:
                        break
                    predict[i + 1] = 0
            
            sample_df.loc[id, 'id'] = id
            for i in range(1, 15):
                sample_df.loc[id, 'p' + str(i)] = predict[i]

            predict_list.append(predict)
    
    return sample_df

sample_df = pd.read_csv(cfg.sample_submission_path)
sample_df = test(test_dataset, sample_df)
sample_df.to_csv(os.path.join('./', 'submission.csv'), index = False)