import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from config import Config
from tqdm import tqdm
import plotly.express as px
import seaborn as sns
sns.set(style='darkgrid', font_scale=1.4)
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def encoder(df, label_to_encode = ['lettr']):
    for label in label_to_encode:
        label_encode = pd.get_dummies
        label_encode = LabelEncoder()
        label_encode.fit(df[label])
        df[label] = label_encode.transform(df[label])
        
        label_mapping = dict(zip(label_encode.classes_, label_encode.transform(label_encode.classes_)))
        print(f"Mapping for {label}: {label_mapping}")
    return df

def train_data_relation(df, save_path = './image'):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    print(df.head(10))
    print('-' * 50)
    print(df.describe())
    print('-' * 50)
    print(df.info())

    corr = df.corr()
    fig = px.imshow(corr, text_auto = True, aspect = 'auto')
    fig.write_image(os.path.join(save_path, 'correlation_matrix.png'))

    for feature in df:
        if feature == 'lettr':
            continue
        plt.figure(figsize=(10, 6))
        sns.countplot(df.reset_index(drop = True),x = df[feature],hue = df['lettr'])
        plt.savefig(os.path.join(save_path, str(feature) + '.png'))
        plt.clf()

def main():
    cfg = Config
    train_df = pd.read_csv(cfg.train_data_path)
    train_df = encoder(train_df)

    test_df = pd.read_csv(cfg.test_data_path)

    train_features = ['x-bar', 'x-ege', 'x2bar', 'xy2br', 'xybar', 'y-ege', 'y2bar', 'yegvx']
    test_features = ['x-bar', 'x-ege', 'x2bar', 'xy2br', 'xybar', 'y-ege', 'y2bar', 'yegvx']

    x_train,x_valid,y_train,y_valid = train_test_split(train_df[train_features], train_df['lettr'], test_size=0.2, random_state=31)
    # train_data_relation(train_df)
    
    # model = RandomForestClassifier(n_estimators = 200, random_state = 31)
    model = MLPClassifier(hidden_layer_sizes=(64,32,16), max_iter=500,activation = 'relu',solver='adam',random_state=31,early_stopping =True)
    model.fit(x_train, y_train)

    y_pred = model.predict(x_valid)
    score = accuracy_score(y_valid,y_pred)
    print(score)

    
    



    

if __name__ == '__main__':
    main()