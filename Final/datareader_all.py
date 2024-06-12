from config import Config
import pandas as pd
import os
from torch.utils.data import Dataset, DataLoader
import json
from tqdm import tqdm

class Dataset(Dataset):
    def __init__(self, behaviors_path, news_path, embeddings_parh):
        self.behaviors_df = pd.read_csv(behaviors_path, sep='\t')
        self.news_df = pd.read_csv(news_path, sep='\t')
        self.embeddings_df = pd.read_csv(embeddings_parh)

    def __len__(self):
        return len(self.behaviors_df)
    
    def get_news_info(self, news_id):
        news_info = self.news_df[self.news_df['news_id'] == news_id].iloc[0]
        category = news_info['category']
        sub_category = news_info['subcategory']
        title = news_info['title']
        abstract = news_info['abstract']
        url = news_info['URL']

        title_entities = news_info['title_entities']
        if isinstance(title_entities, str):
            WikidataIds = [item["WikidataId"] for item in json.loads(title_entities)]
        else:
            WikidataIds = []
            # print(news_id)
        
        title_embeddings = []
        for WikidataId in WikidataIds:
            embeddings = self.get_embedding(WikidataId)
            if embeddings != 0:
                title_embeddings.append(embeddings)
        
        abstract_entities = news_info['abstract_entities']
        if isinstance(abstract_entities, str):
            WikidataIds = [item["WikidataId"] for item in json.loads(abstract_entities)]
        else:
            WikidataIds = []
            # print(news_id)
        abstract_embeddings = []
        for WikidataId in WikidataIds:
            embeddings = self.get_embedding(WikidataId)
            if embeddings != 0:
                abstract_embeddings.append(embeddings)

        return {
            'category': category, 
            'subcategory': sub_category, 
            'title': title, 
            'abstract': abstract, 
            'url': url, 
            'title embeddings': title_embeddings, 
            'abstract embeddings': abstract_embeddings
        }

    def get_embedding(self, entity_id):
        embedding = self.embeddings_df[self.embeddings_df['id'] == entity_id]['embedding']
        if embedding.empty:
            # print(entity_id)
            return 0
        return embedding.iloc[0]
    
    def __getitem__(self, index):
        id = self.behaviors_df['user_id'][index]
        time = self.behaviors_df['time'][index]
        clicked_news = self.behaviors_df['clicked_news'][index].split()
        clicked_news_infos = []
        for news in clicked_news:
            clicked_news_infos.append(self.get_news_info(news))

        impressions_all = self.behaviors_df['impressions'][index].split()
        impression_news = [impressions_data.split('-')[0] for impressions_data in impressions_all]
        impression_news_infos = []
        for news in impression_news:
            impression_news_infos.append(self.get_news_info(news))
        impression_labels = [impressions_data.split('-')[1] for impressions_data in impressions_all]
        return {
            'id': id,
            'time': time,
            'clicked_news': clicked_news,
            'clicked_news_infos': clicked_news_infos,
            'impression_news': impression_news,
            'impression_news_infos': impression_news_infos,
            'impression_labels': impression_labels
        }
    
    


cfg = Config()
train_dataset = Dataset(cfg.train_behaviors_path, cfg.train_news_path, cfg.train_embeddings_path)