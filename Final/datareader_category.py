from config import Config
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm
from sklearn.preprocessing import LabelEncoder


class Dataset(Dataset):
    def __init__(self, behaviors_path, news_path, embeddings_parh, is_test_model = False):
        self.behaviors_df = pd.read_csv(behaviors_path, sep='\t')
        self.news_df = pd.read_csv(news_path, sep='\t')
        self.embeddings_df = pd.read_csv(embeddings_parh)
        self.is_test_model = is_test_model
        self.categorical_cols = ['category', 'subcategory']
        self.news_df = self.label_encode(self.news_df)

    def __len__(self):
        return len(self.behaviors_df)
    
    def get_news_info(self, news_id):
        news_info = self.news_df[self.news_df['news_id'] == news_id]
        if news_info.empty:
            return {'category': 0, 'subcategory': 0}
        news_info = news_info.iloc[0]
        return {
            'category': news_info['category'], 
            'subcategory': news_info['subcategory']
        }

    def label_encode(self, df):
        for ft in self.categorical_cols:
            le = LabelEncoder()
            self.news_df[ft] = le.fit_transform(self.news_df[ft].astype(str))
        return df
    
    def __getitem__(self, index):
        behavior = self.behaviors_df.iloc[index]
        id = int(behavior['id'])
        clicked_news = behavior['clicked_news'].split()
        clicked_news_num = len(clicked_news)
        clicked_news_infos = [self.get_news_info(news) for news in clicked_news]

        if self.is_test_model:
            impression_news = behavior['impressions'].split()
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

        impressions_all = behavior['impressions'].split()
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