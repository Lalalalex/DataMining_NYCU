from tqdm import tqdm
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.metrics import roc_curve, auc
from datareader_category_for_cnn import Dataset
from config import Config
import pandas as pd
import os
import numpy as np
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score
from ranger21 import Ranger21
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self, alpha = 1, gamma = 2, logits = False, reduce = True):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.logits = logits
        self.reduce = reduce

    def forward(self, inputs, targets):
        if self.logits:
            BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction = 'none')
        else:
            BCE_loss = F.binary_cross_entropy(inputs, targets, reduction = 'none')

        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss

        if self.reduce:
            return torch.mean(F_loss)
        else:
            return F_loss


class NewsRecommendationModel(nn.Module):
    def __init__(self):
        super(NewsRecommendationModel, self).__init__()
        self.news_embedding = nn.Sequential(
            torch.nn.Conv2d(1, 4, kernel_size = (1, 3), stride = 1),
            nn.BatchNorm2d(4),
            nn.ReLU(),
            torch.nn.Conv2d(4, 32, kernel_size = (1, 3), stride = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            torch.nn.Conv2d(32, 16, kernel_size = (1, 3), stride = 1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
        )
        self.avgpool = nn.AdaptiveAvgPool2d(output_size = (4, 4))
        self.classifier = nn.Sequential(
            nn.Linear(in_features = 512, out_features = 15, bias = True),
            nn.Sigmoid()
        )

    def forward(self, clicked_news, impression_news):
        clicked_features = self.news_embedding(clicked_news)
        clicked_features = self.avgpool(clicked_features)
        clicked_features = clicked_features.view(clicked_features.size(0), -1)

        impression_embedding = self.news_embedding(impression_news)
        impression_embedding = self.avgpool(impression_embedding)
        impression_embedding = impression_embedding.view(impression_embedding.size(0), -1)

        output = torch.cat((clicked_features, impression_embedding), 1)
        output = self.classifier(output)
        return output

def train_epoch(model, dataloader, criterion, optimizer, cfg, epoch = 0, class_weights = torch.tensor([0.999, 0.001])):
    print('Epoch ' + str(epoch) + ':')
    model.train()
    total_loss = 0
    total_auc = 0
    with tqdm(dataloader, unit = 'batch', desc = 'Train') as tqdm_loader:
        for index, data in enumerate(tqdm_loader):
            clicked_news = data['clicked_news_categories'].clone().detach().to(dtype = torch.float, device = cfg.device)
            impression_news = data['impression_news_categories'].clone().detach().to(dtype = torch.float, device = cfg.device)
            label = torch.tensor(data['impression_labels']).to(dtype = torch.float, device=cfg.device)

            predict = model(clicked_news, impression_news)
            predict = predict.reshape(-1)
            loss = criterion(predict, label)
            predict = torch.round(predict).cpu().detach()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            loss = loss.detach().item()
            total_loss = total_loss + loss

            fpr, tpr, _ = roc_curve(label.cpu(), predict)
            auc_score = auc(fpr, tpr)

            total_auc = total_auc + auc_score
            average_auc = total_auc/(index + 1)
            average_loss = total_loss/(index + 1)

            tqdm_loader.set_postfix(loss = loss, average_loss = average_loss, auc = auc_score, average_auc = average_auc, predict = predict)


cfg = Config()
train_dataset = Dataset(cfg.train_behaviors_path, cfg.train_news_path, cfg.train_embeddings_path)
train_dataloader = DataLoader(train_dataset, batch_size = cfg.batch_size, num_workers = 4, drop_last = False)
model = NewsRecommendationModel()
model = model.to(device = cfg.device)
criterion = nn.BCELoss()
optimizer = Ranger21(model.parameters(), lr = cfg.learning_rate, num_epochs = cfg.epochs, num_batches_per_epoch = len(train_dataset))
train_epoch(model, train_dataloader, criterion, optimizer, cfg)