import numpy as np
import pandas as pd
import os

class Config():
    base_path = '/home/hyhsieh/DM/final_data'
    base_data_path = os.path.join(base_path, 'data')
    train_path = os.path.join(base_data_path, 'train')
    train_behaviors_path = os.path.join(train_path, 'train_behaviors.tsv')
    train_news_path = os.path.join(train_path, 'train_news.tsv')
    train_embeddings_path = os.path.join(train_path, 'train_entity_embedding.csv')
    test_path = os.path.join(base_data_path, 'test')
    test_behaviors_path = os.path.join(test_path, 'test_behaviors.tsv')
    test_news_path = os.path.join(test_path, 'test_news.tsv')
    test_embeddings_path = os.path.join(test_path, 'test_entity_embedding.csv')
    sample_submission_path = os.path.join(base_data_path, 'sample_submission.csv')

    output_path = './bert'
    batch_size = 500
    test = False
    model_weight = './bert/best_lstm_model.pth'
    num_epochs = 500
    learning_rate = 1e-4
    
