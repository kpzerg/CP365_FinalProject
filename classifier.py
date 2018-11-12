import numpy as np
import pandas as pd
#import re
from scipy.sparse import csr_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neural_network import MLPClassifier

data = pd.read_csv('formatted_data.csv')

X = csr_matrix(TfidfVectorizer().fit_transform(data.values[:, 0].astype('U')))
y = data.values(:, 1].astype('int') 

X_train = X[:-100]
y_train = y[:-100]
X_test = X[-100:]
y_test = y[-100:]

regressors = []
regressors.append(MLPClassifier())
regressors.append(MLPClassifier(activation='logistic'))
regressors.append(MLPClassifier(activation='tanh'))

for regressor in regressors:
    regressor.fit(X_train, y_train)
    predicted = regressor.predict(X_test)
    
    print(regressor.score(X_test))
