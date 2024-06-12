import pandas as pd
import os
import numpy as np
from tqdm import tqdm

base_path = '/home/twhuang/homework/Data_Mining/HW1'
train_df = pd.read_csv(os.path.join(base_path, 'train.csv'))

train_data = {}

def get_data(data):
    try:
        int(data)
        return int(data)
    except ValueError:
        return -1

def compute_mse(prediction, ground_truth):
    return np.sum((ground_truth - prediction)**2) / len(prediction)

train_data_list = []

index = 0
while(index < len(train_df)):
    data = {}
    data['location'] = train_df.iloc[index]['Location']
    data['date'] = train_df.iloc[index]['Date']
    for i in range(10):
        if get_data(train_df.iloc[index + 9][str(i)]) == -1 and i == 0:
            data['PM2.5 ' + str(i)] = 0
        elif get_data(train_df.iloc[index + 9][str(i)]) == -1:
            data['PM2.5 ' + str(i)] = data['PM2.5 ' + str(i - 1)]
        else:
            data['PM2.5 ' + str(i)] = get_data(train_df.iloc[index + 9][str(i)])
    index = index + 18
    train_data_list.append(data)

#print(data_list)

input = np.zeros((len(train_data_list), 9))
label = np.zeros(len(train_data_list))

for i in range(input.shape[0]):
    for j in range(len(input[i] - 1)):
        input[i][j] = train_data_list[i]['PM2.5 ' +str(j)]
    label[i] = train_data_list[i]['PM2.5 9']


class LinearRegressionCloseform():
    def __init__(self):
        self.weights = None
        self.intercept = None
    def fit(self, X, y, regulariztion_rate: int = 100):
        # if regulariztion_rate = 0, then not use L1 loss
        A = np.insert(X, len(X.T), [1], axis=1)
        # A = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A)), A.T), y)
        A = np.matmul(np.matmul(np.linalg.inv(np.matmul(A.T, A) + regulariztion_rate * np.identity(len(A.T))), A.T), y)
        self.weights = A[:len(X.T)]
        self.intercept = A[-1]

    def predict(self, X):
        return np.matmul(X, self.weights) + self.intercept

linear_regression = LinearRegressionCloseform()
linear_regression.fit(input, label)
predict = linear_regression.predict(input)

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

test_df = pd.read_csv(os.path.join(base_path, 'test.csv'))

test_data_list = []
index = 0
while(index < len(test_df)):
    data = {}    
    for i in range(2, 11):
        if get_data(test_df.iloc[index + 8][i]) == -1 and i == 0:
            data['PM2.5 ' + str(i - 2)] = 0
        elif get_data(test_df.iloc[index + 8][i]) == -1:
            data['PM2.5 ' + str(i - 2)] = data['PM2.5 ' + str(i - 1)]
        else:
            data['PM2.5 ' + str(i - 2)] = get_data(test_df.iloc[index + 8][i])
    index = index + 18
    test_data_list.append(data)

test_input = np.zeros((len(test_data_list), 9))
for i in range(test_input.shape[0]):
    for j in range(len(test_input[i] - 1)):
        test_input[i][j] = test_data_list[i]['PM2.5 ' +str(j)]
#print(test_data_list)

predict = linear_regression.predict(test_input)

index_list = []
predict_list = []

for i in range(len(predict)):
    index_list.append('index_' + str(i))
    predict_list.append(predict[i])

sample_df = pd.read_csv(os.path.join(base_path, 'sample submission.csv'))
submit_df  = pd.DataFrame(columns = sample_df.columns)
submit_df['index'] = index_list
submit_df['answer']=predict_list
submit_df.to_csv(os.path.join('./', 'submission.csv'),index=False)