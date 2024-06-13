from config import Config
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder
import numpy as np
from collections import Counter


class Dataset(Dataset):
    def __init__(self, behaviors_path, news_path, embeddings_parh, is_test_model = False, encode_type = 'one_hot'):
        self.behaviors_df = pd.read_csv(behaviors_path, sep='\t')
        self.news_df = pd.read_csv(news_path, sep='\t')
        self.embeddings_df = pd.read_csv(embeddings_parh)
        self.is_test_model = is_test_model
        self.encode_type = encode_type
        self.categorical_cols = ['category', 'subcategory']
        if encode_type == 'one_hot':
            self.news_df = self.one_hot_encode(self.news_df)
        else:
            self.news_df = self.label_encode(self.news_df)

    def __len__(self):
        return len(self.behaviors_df)
    
    def get_news_info(self, news_id):
        news_info = self.news_df[self.news_df['news_id'] == news_id]
        if self.encode_type == 'label':
            if news_info.empty:
                return {'category': 0, 'subcategory': 0}
            news_info = news_info.iloc[0]
            return {
                'category': news_info['category'], 
                'subcategory': news_info['subcategory']
            }
        
        if news_info.empty:
            return {'category': np.zeros(len(self.news_df.filter(like='category_').columns)), 
                    'subcategory': np.zeros(len(self.news_df.filter(like='subcategory_').columns))}
        news_info = news_info.iloc[0]
        return {
            'category': news_info.filter(like='category_').astype(int).to_numpy(),
            'subcategory': news_info.filter(like='subcategory_').astype(int).to_numpy()
        }

    def label_encode(self, df):
        for ft in self.categorical_cols:
            le = LabelEncoder()
            self.news_df[ft] = le.fit_transform(self.news_df[ft].astype(str))
        return df
    
    def one_hot_encode(self, df):
        for ft in self.categorical_cols:
            df = pd.get_dummies(df, columns=[ft], prefix=[ft])
        return df
    
    def get_max_user_clicked_news_count(self):
        self.behaviors_df['clicked_news_count'] = self.behaviors_df['clicked_news'].apply(lambda x: len(x.split()))
        max_clicked_news_count = self.behaviors_df['clicked_news_count'].max()
        return max_clicked_news_count
    
    def __getitem__(self, index):
        behavior = self.behaviors_df.iloc[index]
        id = int(behavior['id'])
        clicked_news = behavior['clicked_news'].split()
        clicked_news_num = len(clicked_news)
        clicked_news_infos = [self.get_news_info(news) for news in clicked_news]

        if self.encode_type == 'label':
            clicked_news_categories = np.concatenate([info['category'] for info in clicked_news_infos])
            clicked_news_categories = np.expand_dims(clicked_news_categories, axis = 0)
            clicked_news_subcategories = np.concatenate([info['subcategory'] for info in clicked_news_infos])
            clicked_news_subcategories = np.expand_dims(clicked_news_subcategories, axis = 0)
        else:
            clicked_news_categories = np.vstack([info['category'] for info in clicked_news_infos])
            clicked_news_categories = np.expand_dims(clicked_news_categories, axis = 0)
            clicked_news_subcategories = np.vstack([info['subcategory'] for info in clicked_news_infos])
            clicked_news_subcategories = np.expand_dims(clicked_news_subcategories, axis = 0)

        if self.is_test_model:
            impression_news = behavior['impressions'].split()
            impression_news_infos = [self.get_news_info(news_id) for news_id in impression_news]
            impression_news_categories = np.vstack([info['category'] for info in impression_news_infos])
            impression_news_categories = np.expand_dims(impression_news_categories, axis = 0)
            impressio_news_subcategories = np.vstack([info['subcategory'] for info in impression_news_infos])
            impressio_news_subcategories = np.expand_dims(impressio_news_subcategories, axis = 0)
            
            return {
                'id': id,
                'clicked_news': clicked_news,
                'clicked_news_categories': clicked_news_categories,
                'clicked_news_subcategories': clicked_news_subcategories,
                'impression_news': impression_news,
                'impression_news_categories': impression_news_categories,
                'impression_news_subcategories': impressio_news_subcategories,
                'clicked_news_num': clicked_news_num
            }

        impressions_all = behavior['impressions'].split()
        impression_news = [impressions_data.split('-')[0] for impressions_data in impressions_all]
        impression_news_infos = [self.get_news_info(news_id) for news_id in impression_news]
        impression_news_categories = np.vstack([info['category'] for info in impression_news_infos])
        impression_news_categories = np.expand_dims(impression_news_categories, axis = 0)
        impressio_news_subcategories = np.vstack([info['subcategory'] for info in impression_news_infos])
        impressio_news_subcategories = np.expand_dims(impressio_news_subcategories, axis = 0)
        impression_labels = [int(impressions_data.split('-')[1]) for impressions_data in impressions_all]

        return {
            'id': id,
            'clicked_news': clicked_news,
            'clicked_news_categories': clicked_news_categories,
            'clicked_news_subcategories': clicked_news_subcategories,
            'impression_news': impression_news,
            'impression_news_categories': impression_news_categories,
            'impression_news_subcategories': impressio_news_subcategories,
            'impression_labels': impression_labels,
            'clicked_news_num': clicked_news_num
        }