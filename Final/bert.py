import os
import torch
import logging
import json
import shutil
import pickle
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_auc_score
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader, Dataset, random_split

from utils import set_logger
from config import Config

MAX_CLICKED_LEN = 35
BERT_EMBEDDING_DIM = 768

class NewsRecommendationDataset(Dataset):
    def __init__(self, base_root, history_seq_len=35, is_train=True):
        if is_train:
            mode = 'train'
        else:
            mode = 'test'
        behaviors_path = os.path.join(base_root, f'{mode}_behaviors.tsv')
        news_path = os.path.join(base_root, f'{mode}_news.tsv')
        embeddings_path = os.path.join(base_root, f'{mode}_entity_embedding.csv')

        self.behaviors_df = pd.read_csv(behaviors_path, sep='\t')
        self.news_df = pd.read_csv(news_path, sep='\t')
        self.embeddings_df = pd.read_csv(embeddings_path)

        # Load pre-trained BERT tokenizer and model
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.encoder_bert = BertModel.from_pretrained('bert-base-uncased')

        self.is_train = is_train

        dict_path = os.path.join(base_root, f"{mode}_news_embeddings_dict.pkl")
        if not os.path.exists(dict_path):
            dict_path = None

        # step1: encode news to embeddings
        self.encoded_news_dict = self.news_encoding(dict_path) # key-value: news_id-embedding vector

        self.history_seq_len = history_seq_len


    def __len__(self):
        return len(self.behaviors_df)

    def __getitem__(self, index):

        # history
        clicked_news = self.behaviors_df['clicked_news'][index].split()
        clicked_news_embeddings = self.get_news_list_embeddings(clicked_news)
        clicked_news_embeddings = clicked_news_embeddings.reshape(-1, clicked_news_embeddings.shape[2])
        # Padding or truncating for lstm batch training
        history_embeddings = np.zeros((self.history_seq_len, BERT_EMBEDDING_DIM))
        min_len = min(self.history_seq_len, clicked_news_embeddings.shape[0])
        history_embeddings[:min_len][:BERT_EMBEDDING_DIM] = clicked_news_embeddings[:min_len][:BERT_EMBEDDING_DIM]

        # impression
        impressions_all = self.behaviors_df['impressions'][index].split()
        impression_news = [impressions_data.split('-')[0] for impressions_data in impressions_all]
        impression_news_embeddings = self.get_news_list_embeddings(impression_news)
        impression_news_embeddings = impression_news_embeddings.reshape(-1, impression_news_embeddings.shape[2])
        if self.is_train:
            impression_labels = [float(impressions_data.split('-')[1]) for impressions_data in impressions_all]
        else:
            impression_labels = []

        return history_embeddings, impression_news_embeddings, np.array(impression_labels)

    def get_news_embeddings(self, news_id):

        if news_id not in self.encoded_news_dict:
            return torch.zeros((1, 768))
        return self.encoded_news_dict[news_id]

    def get_news_list_embeddings(self, news_list):

        news_embeddings = []
        for news in news_list:
            news_embeddings.append(self.get_news_embeddings(news))
        return np.array(news_embeddings)

    def news_encoding(self, file_path):
        if file_path is not None:
            # 加载字典从文件
            with open(file_path, 'rb') as file:
                loaded_embeddings_dict = pickle.load(file)
            return loaded_embeddings_dict

        columns_to_keep = ['news_id', 'category', 'subcategory', 'title', 'abstract']
        df = self.news_df[columns_to_keep].copy()
        # 確保所有列都是字符串類型
        df['category'] = df['category'].astype(str)
        df['subcategory'] = df['subcategory'].astype(str)
        df['title'] = df['title'].astype(str)
        df['abstract'] = df['abstract'].astype(str)

        # 將DataFrame中的列合併
        df['combined_text'] = df[['category', 'subcategory', 'title', 'abstract']].apply(lambda x: ' '.join(x), axis=1)

        # 獲取每篇文章的BERT嵌入表示
        embeddings_dict = {}
        for index, row in tqdm(df.iterrows(), total=df.shape[0], desc="Dataset news encoding"):
            news_id = row['news_id']
            combined_text = row['combined_text']
            embeddings_dict[news_id] = self.get_bert_embeddings(combined_text)

        if self.is_train:
            dict_name = 'train_news_embeddings_dict.pkl'
        else:
            dict_name = 'test_news_embeddings_dict.pkl'

        with open(dict_name, 'wb') as file:
            pickle.dump(embeddings_dict, file)

        return embeddings_dict

    # Function to get BERT embeddings
    def get_bert_embeddings(self, text):
        # Tokenize input text
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
        
        # Get hidden states from the BERT model
        with torch.no_grad():
            outputs = self.encoder_bert(**inputs)
        
        # The embeddings are in the last hidden state
        embeddings = outputs.last_hidden_state
        
        # Mean pooling of the token embeddings to get a single vector representation
        pooled_embeddings = torch.mean(embeddings, dim=1)
        
        return pooled_embeddings
    

class NewsRecommendationModel(nn.Module):
    def __init__(self, embedding_dim=768, hidden_dim=384, output_dim=768):
        super(NewsRecommendationModel, self).__init__()

        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim * MAX_CLICKED_LEN, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x_clicked):

        lstm_out, _ = self.lstm(x_clicked)
        flat = lstm_out.reshape(lstm_out.size(0), -1)
        output = self.fc(flat)
        output = self.sigmoid(output)
        return output

class DeepLearning(object):
    def __init__(self, model, optimizer, criterion, device):
        self.optim = optimizer
        self.criterion = criterion
        self.device = device
        self.model = model
        self.metrics = {'AUC': roc_auc_score}


    def train(self, train_loader, val_loader, args):
        best_auc = 0
        for epoch in range(args.num_epochs):
            self.model.train()
            epoch_loss, epoch_acc = self.train_one_epoch(train_loader, epoch+1)
            val_loss, val_acc = self.validation(val_loader)

            logging.info(f'Epoch {epoch+1}/{args.num_epochs}: AVG Loss={epoch_loss}, AUC={epoch_acc} Validation: Loss={val_loss}, AUC={val_acc}')
            torch.save(self.model.state_dict(), os.path.join(args.output_path, f"lstm_model.pth"))

            if val_acc > best_auc:
                best_auc = val_acc
                torch.save(self.model.state_dict(), os.path.join(args.output_path, f"best_lstm_model.pth"))

    def train_one_epoch(self, train_loader, step):
        with tqdm(train_loader, unit="batch") as tepoch:
            total_loss = []
            total_auc = []
            for X_clicked_batch, X_impressions_batch, y_batch in tepoch:
                tepoch.set_description(f"Train EPOCH{step}")

                X_clicked_batch = X_clicked_batch.float().to(self.device)
                X_impressions_batch = X_impressions_batch.float().to(self.device)
                y_batch = y_batch.to(self.device)

                self.optim.zero_grad()
                output = self.model(X_clicked_batch)

                user_embeddings = output.unsqueeze(-1)  # 变为 (batch_size, 768, 1)
                dot_product = torch.matmul(X_impressions_batch, user_embeddings).squeeze(-1)  # 结果形状为 (batch_size, 15)

                loss = self.criterion(dot_product, y_batch)
                loss.backward()
                self.optim.step()
                metric = self.metrics['AUC']

                try:
                    auc = metric(y_batch.cpu().detach().numpy(), dot_product.cpu().detach().numpy()).item()
                except ValueError as e:
                    # Handle the case where ROC AUC cannot be calculated
                    auc = 0
                
                tepoch.set_postfix(loss=loss.item(), auc=auc)
                total_loss.append(loss.item())
                total_auc.append(auc)
            return sum(total_loss) / len(total_loss), sum(total_auc) / len(total_auc)
        

    def validation(self, val_loader):
        self.model.eval()
        with torch.no_grad():
            with tqdm(val_loader, unit="batch") as tepoch:
                total_loss = []
                total_auc = []
                for X_clicked_batch, X_impressions_batch, y_batch in tepoch:
                    tepoch.set_description(f"Validation")

                    X_clicked_batch = X_clicked_batch.float().to(self.device)
                    X_impressions_batch = X_impressions_batch.float().to(self.device)
                    y_batch = y_batch.to(self.device)

                    output = self.model(X_clicked_batch)
                    user_embeddings = output.unsqueeze(-1)  # 变为 (batch_size, 768, 1)
                    dot_product = torch.matmul(X_impressions_batch, user_embeddings).squeeze(-1)  # 结果形状为 (batch_size, 15)

                    loss = self.criterion(dot_product, y_batch)
                    metric = self.metrics['AUC']

                    try:
                        auc = metric(y_batch.cpu().detach().numpy(), dot_product.cpu().detach().numpy()).item()
                    except ValueError as e:
                        # Handle the case where ROC AUC cannot be calculated
                        auc = 0
     
                    tepoch.set_postfix(loss=loss.item(), auc=auc)
                    total_loss.append(loss.item())
                    total_auc.append(auc)

                return sum(total_loss) / len(total_loss), sum(total_auc) / len(total_auc)
    
    def test(self, test_loader):
        self.model.eval()
        with torch.no_grad():
            predictions = []
            with tqdm(test_loader, unit="batch") as tepoch:

                for X_clicked_batch, X_impressions_batch, _ in tepoch:
                    tepoch.set_description(f"Test")

                    X_clicked_batch = X_clicked_batch.float().to(self.device)
                    X_impressions_batch = X_impressions_batch.float().to(self.device)

                    output = self.model(X_clicked_batch)
                    user_embeddings = output.unsqueeze(-1)  # 变为 (batch_size, 768, 1)
                    dot_product = torch.matmul(X_impressions_batch, user_embeddings).squeeze(-1)  # 结果形状为 (batch_size, 15)
                    predictions.append(dot_product.cpu().numpy())
                result = np.concatenate(predictions, axis=0)
                print(result.shape)

                return result

def output_submission_csv(pred):
    # replace column p1 to p15 with prediction results
    df_pred = pd.DataFrame(pd.read_csv("submission.csv"))
    df_pred.iloc[:pred.shape[0], 1:pred.shape[1]+1] = pred
    output_file_name = f"output.csv"
    df_pred.to_csv(output_file_name, index = False)
    print(f"Write successfully to {output_file_name}")


def main():
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using {device}")

    # 構建DataLoader
    train_dataset = NewsRecommendationDataset(cfg.train_path, MAX_CLICKED_LEN)
    generator1 = torch.Generator().manual_seed(42)
    train_set, val_set = random_split(train_dataset, [0.8, 0.2], generator1)
    test_dataset = NewsRecommendationDataset(cfg.test_path, MAX_CLICKED_LEN, is_train=False)
    print(f"train dataset length: {len(train_set)}")
    print(f"validation dataset length: {len(val_set)}")
    print(f"test dataset length: {len(test_dataset)}")

    train_dataloader = DataLoader(train_set, batch_size=cfg.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_set, batch_size=cfg.batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)

    for data in train_dataloader:
        history, impression, lable =  data
        print(f"history: {history.shape}")
        print(f"impression: {impression.shape}")
        print(f"lable: {lable}")
        break

    lstm_model = NewsRecommendationModel().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(lstm_model.parameters(), lr=cfg.learning_rate)
    deep_learning = DeepLearning(lstm_model, optimizer, criterion, device)

    if cfg.test:
        deep_learning.model.load_state_dict(torch.load(cfg.model_weight))
        print(f"Load model weights from {cfg.model_weight}")
        predictions = deep_learning.test(test_dataloader)
        output_submission_csv(predictions)
        return

    # make output folder
    Path(cfg.output_path).mkdir(parents=True, exist_ok=True)
    set_logger(os.path.join(cfg.output_path, 'train.log'))

    shutil.copy('./config.py', os.path.join(cfg.output_path, 'train_config.py'))
    logging.info("Start Training")
    # 訓練模型
    deep_learning.train(train_dataloader, val_dataloader, cfg)


    
if __name__ == '__main__':
    main()
