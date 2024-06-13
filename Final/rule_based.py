from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_curve, auc
from datareader_category import Dataset
from config import Config
import pandas as pd
import os

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
            
            fpr, tpr, _ = roc_curve(label, predict)
            accuracy = auc(fpr, tpr)
            total_accuracy = total_accuracy + accuracy
            average_accuracy = total_accuracy/(index + 1)
            tqdm_loader.set_postfix(average_accuracy = average_accuracy, accuracy = accuracy)

def test(dataset, sample_df):
    predict_list = []
    with tqdm(dataset, unit = 'data', desc = 'Test') as tqdm_loader:
        for index, data in enumerate(tqdm_loader):
            id = data['id']
            clicked_news_infos = data['clicked_news_infos']
            impression_news_infos = data['impression_news_infos']

            predict = [1 for i in range(len(impression_news_infos))]

            for i, impression_news in enumerate(impression_news_infos):
                for clicked_news in clicked_news_infos:
                    if clicked_news['category'] == impression_news['category']:
                        break
                    predict[i] = 0
            
            sample_df.loc[index, 'id'] = id
            for i in range(15):
                sample_df.loc[index, 'p' + str(i + 1)] = predict[i]

            predict_list.append(predict)
    
    return sample_df


cfg = Config()
train_dataset = Dataset(cfg.train_behaviors_path, cfg.train_news_path, cfg.train_embeddings_path)
test_dataset = Dataset(cfg.test_behaviors_path, cfg.test_news_path, cfg.test_embeddings_path, is_test_model = True)
sample_df = pd.read_csv(cfg.sample_submission_path)
sample_df = test(test_dataset, sample_df)
sample_df.to_csv(os.path.join('./', 'submission.csv'), index = False)