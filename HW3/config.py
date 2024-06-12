import os

class Config:
    base_path = '/home/twhuang/homework/Data_Mining/HW3'
    data_path = os.path.join(base_path, 'data')
    train_data_path = os.path.join(data_path, 'training.csv')
    test_data_path = os.path.join(data_path, 'test_X.csv')
    sample_path = os.path.join(data_path, 'sample.csv')