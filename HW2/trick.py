import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
import numpy as np

def show_curve(data, file_name = 'learning_curve', title = 'Learning Curve', x = 'epoch', y = 'accuracy'):
    plt.figure()
    plt.title(title)
    for i in data:
        plt.plot(data[i], label = i)
    plt.xlabel(x)
    plt.ylabel(y)
    plt.legend()
    plt.savefig(file_name)

title_train_acc = [0.36, 0.42, 0.48, 0.54, 0.61, 0.65, 0.67, 0.71, 0.74, 0.76]
title_valid_acc = [0.38, 0.46, 0.48, 0.46, 0.51, 0.49, 0.512, 0.53, 0.51, 0.52]

text_train_acc = [0.36, 0.42, 0.21, 0.16, 0.13, 0.11, 0.09, 0.10, 0.11, 0.11]
text_valid_acc = [0.32, 0.36, 0.16, 0.12, 0.07, 0.08, 0.076, 0.09, 0.10, 0.11]

title_text_train_acc = [0.28, 0.32, 0.36, 0.44, 0.48, 0.54, 0.61, 0.67, 0.64, 0.71]
title_text_valid_acc = [0.34, 0.40, 0.42, 0.48, 0.51, 0.54, 0.58, 0.61, 0.59, 0.61]

data = {
    'title_train': title_train_acc,
    'title_valid': title_valid_acc,
    'text_train': text_train_acc,
    'text_valid': text_valid_acc,
    'titla_text_train': title_text_train_acc,
    'title_text_valid': title_text_valid_acc
}

show_curve(data)