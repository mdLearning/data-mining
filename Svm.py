import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn import datasets
from sklearn import svm

if __name__ == '__main__':
    print('prepare datasets...')

    #minist数据集
    raw_data = pd.read_csv('./data/train_binary.csv', header=0)
    data = raw_data.values
    features = data[::, 1::]
    labels = data[::, 0]

    train_features,test_features,train_labels,test_labels = train_test_split(features,labels,test_size=0.33,random_state=0)

    time_2 = time.time()
    print('Start training...')
    clf = svm.SVC()#svm class
    clf.fit(train_features,train_labels)
    time_3 = time.time()
    print('training cost %d seconds' % (time_3 - time_2))

    print('Start predict...')
    test_predict = clf.predict(test_features)

    time_4 = time.time()

    print('predicting cost %d seconds' % (time_4 - time_3))

    score = accuracy_score(test_labels, test_predict)
    print('The accuracy score is %f' % score)