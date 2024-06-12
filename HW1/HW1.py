import pandas as pd
import os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

base_path = '/home/twhuang/homework/Data_Mining/HW1'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))
test_df = pd.read_csv(os.path.join(base_path, 'test.csv'), header=None)
sample_df = pd.read_csv(os.path.join(base_path, 'sample submission.csv'))
submit_df  = pd.DataFrame(columns = sample_df.columns)

def show_curve(loss, file_name = 'loss', title = 'Loss Curve', x = 'epoch', y = 'loss'):
    plt.figure()
    plt.title(title)
    plt.plot(loss)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.savefig(file_name)

class LinearRegressionCloseform():
    def __init__(self):
        self.weights = None
        self.intercept = None
    def fit(self, X, y, regulariztion_rate: int = 1):
        A = np.insert(X, len(X.T), [1], axis=1)
        A = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A) + regulariztion_rate * np.identity(len(A.T))), A.T), y)
        self.weights = A[:len(X.T)]
        self.intercept = A[-1]

    def predict(self, X):
        return np.round(np.matmul(X, self.weights) + self.intercept, 0)

def get_data(data):
    try:
        float(data)
        return float(data)
    except ValueError:
        return -1

class LinearRegressionGradientdescent():
    def __init__(self):
        self.weights = None
        self.intercept = None
        self.loss = []
    def fit(self, X, y, learning_rate: float = 1e-4, epochs: int = 100000, regulariztion_rate: int = 1):
        y = np.squeeze(y)
        self.weights = np.random.normal(0, 1, (len(X.T)))
        self.intercept = np.random.normal(0, 1, 1)[0]
        losses = []
        self.m_w = np.zeros(len(X.T))
        self.v_w = np.zeros(len(X.T))
        self.m_i = 0
        self.v_i = 0
        beta = 0.9
        with tqdm(range(epochs), unit = 'Step', desc = 'Train') as tqdm_loader:
            for epoch in tqdm_loader:
                prediction = np.round(np.matmul(X, self.weights) + self.intercept)
                loss = compute_mse(prediction, y)
                weights_gradient = 2 * np.matmul(X.T, prediction - y) / len(X) + 2 * regulariztion_rate * self.weights
                intercept_gradient = 2 * np.mean(prediction - y)
                self.m_w = beta * self.m_w + (1 - beta) * weights_gradient
                self.v_w = beta * self.v_w + (1 - beta) * (weights_gradient ** 2)
                m_hat = self.m_w / (1 - beta)
                v_hat = self.v_w / (1 - beta)
                self.weights = self.weights - (learning_rate * m_hat  / np.sqrt(v_hat + 1e-5))
                self.m_i = beta * self.m_i + (1 - beta) * intercept_gradient
                self.v_i = beta * self.v_i + (1 - beta) * (intercept_gradient ** 2)
                m_hat = self.m_i / (1 - beta)
                v_hat = self.v_i / (1 - beta)
                self.intercept = self.intercept - (learning_rate * m_hat  / np.sqrt(v_hat + 1e-5))
                losses.append(loss)
                tqdm_loader.set_postfix(loss = loss)
                if epoch > 50000:
                    self.loss.append(loss)
        return losses

    def predict(self, X):
        return np.round(np.matmul(X, self.weights) + self.intercept)


def compute_mse(prediction, ground_truth):
    return np.sum((ground_truth - prediction)**2) / len(prediction)

input_list = []
label_list = []
item_all = [8, 9]
item_normalize = [0, 1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13, 14, 15, 16, 17]
with tqdm(range(18), unit = 'item', desc = 'Train Normalize'):
    for item in range(18):
        if item not in item_normalize:
            break
        data_max = 0
        data_min = 10000
        for index in range(0, len(train_df), 18):
            for hour in range(9):
                value = get_data(train_df.iloc[index + item][str(hour)])
                if value != -1:
                    data_max = max(value, data_max)
                    data_min = min(value, data_min)
        for index in range(0, len(train_df), 18):
            for hour in range(9):
                if get_data(train_df.iloc[index + item][str(hour)]) != -1:
                    train_df.iloc[index + item][str(hour)] = (get_data(train_df.iloc[index + item][str(hour)]) - data_min)/(data_max - data_min)

train_df.to_csv(os.path.join('./', 'train_normalize.csv'),index = False)

with tqdm(range(18), unit = 'item', desc = 'Test Normalize'):
    for item in range(18):
        if item not in item_normalize:
            break
        data_max = 0
        data_min = 10000
        for index in range(0, len(test_df), 18):
            for hour in range(2, 11):
                value = get_data(test_df.iloc[index + item][hour])
                if value != -1:
                    data_max = max(value, data_max)
                    data_min = min(value, data_min)
        for index in range(0, len(test_df), 18):
            for hour in range(2, 11):
                if get_data(test_df.iloc[index + item][hour]) != -1:
                    test_df.iloc[index + item][hour] = (get_data(test_df.iloc[index + item][hour]) - data_min)/(data_max - data_min)

test_df.to_csv(os.path.join('./', 'test_normalize.csv'),index = False)

for index in range(0, len(train_df), 18):
    data = []
    for item in range(18):
        if item not in item_all:
            total_data = 0.0
            for hour in range(9):
                value = get_data(train_df.iloc[index + item][str(hour)])
                if value != -1:
                    total_data = total_data + value
            data.append(total_data / 9)
        else:
            for hour in range(9):
                value = get_data(train_df.iloc[index + item][str(hour)])
                if value != -1:
                    data.append(value)
                elif hour == 0:
                    data.append(0.0)
                else:
                    data.append(get_data(train_df.iloc[index + item][str(hour - 1)]))
    label = get_data(train_df.iloc[index + 9][str(9)])
    if label != -1:
        label_list.append(int(label))
    else:
        label_list.append(0)
    input_list.append(data)

input_data = np.array(input_list)
label = np.array(label_list)

model =  LinearRegressionGradientdescent()
model.fit(input_data, label, regulariztion_rate = 0.1)
predict = model.predict(input_data)
total_error = 0
for i in range(len(predict)):
    total_error = total_error + abs(predict[i] - label[i])
average_error = total_error/len(predict)
print(average_error)

index_list = []
predict_list = []
label_list = []

for i in range(len(predict)):
    index_list.append('index_' + str(i))
    predict_list.append(predict[i])
    label_list.append(label[i])

train_df = pd.DataFrame()
train_df['index'] = index_list
train_df['predict'] = predict_list
train_df['answer'] = label_list
train_df.to_csv(os.path.join('./', 'trains_predict.csv'),index=False)


test_input_list = []
for index in range(0, len(test_df), 18):
    data = []
    for item in range(18):
        if item not in item_all:
            total_data = 0.0
            for hour in range(2, 11):
                value = get_data(test_df.iloc[index + item][hour])
                if value != -1:
                    total_data = total_data + value
            data.append(total_data / 9)
        else:
            for hour in range(2, 11):
                value = get_data(test_df.iloc[index + item][hour])
                if value != -1:
                    data.append(value)
                elif hour == 0:
                    data.append(0.0)
                else:
                    data.append(get_data(test_df.iloc[index + item][hour - 1]))
    test_input_list.append(data)

test_input_data = np.array(test_input_list)
predict = model.predict(test_input_data)

index_list = []
predict_list = []

for i in range(len(predict)):
    index_list.append('index_' + str(i))
    predict_list.append(predict[i])

sample_df = pd.read_csv(os.path.join(base_path, 'sample submission.csv'))
submit_df  = pd.DataFrame(columns = sample_df.columns)
submit_df['index'] = index_list
submit_df['answer']=predict_list
submit_df.to_csv(os.path.join('./', 'submission.csv'),index = False)