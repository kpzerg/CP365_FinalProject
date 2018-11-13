import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('formatted_data.csv')

vocab = np.unique(sum([sentence.split() for sentence in data.values[:, 0].astype(str)], []))

vectorizer = TfidfVectorizer(vocabulary=vocab)

X = csr_matrix(vectorizer.fit_transform(data.values[:, 0].astype('U')))
y = data.values[:, 1].astype('int') 

X_train = X[:-100]
y_train = y[:-100]
X_test = X[-100:]
y_test = y[-100:]

regressors = []
regressors.append(MLPClassifier())
regressors.append(MLPClassifier(activation='logistic'))
regressors.append(MLPClassifier(activation='tanh'))

for regressor in regressors:
    print('fitting for classifer %s' % (regressor))
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    pos_pos = 0
    pos_neg = 0
    neg_pos = 0
    neg_neg = 0

    for x, y in zip(y_test, predicted):
        if x and y:
            pos_pos += 1
        elif x and not y:
            pos_neg += 1
        elif not x and y:
            neg_pos += 1
        else:
            neg_neg += 1
    
    print("%s %s %s %s" % (pos_pos, pos_neg, neg_pos, neg_neg))
    print(regressor.score(X_test, y_test))
