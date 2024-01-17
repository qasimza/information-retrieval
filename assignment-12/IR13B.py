# IR13B.py CS5154/6054 cheng 2022
# BernoulliNB classification
# Usage: python IR13B.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import accuracy_score

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N =len(docs)
trainX = np.concatenate([docs[0:1000], docs[N-1000:N]])
y = np.concatenate([np.zeros(1000, dtype=np.int16), np.ones(1000, dtype=np.int16)])
testX = np.concatenate([docs[1000:1100], docs[N-1100:N-1000]])
testY = np.concatenate([np.zeros(100, dtype=np.int16), np.ones(100, dtype=np.int16)])

cv = CountVectorizer(binary=True, max_df=0.4, min_df=4)
X = cv.fit_transform(trainX).toarray()
print(X.shape)
voc = cv.get_feature_names()
T = cv.transform(testX).toarray()

model = BernoulliNB()
model.fit(X, y)
pred = model.predict(T)
print(pred)
print ('Accuracy Score - ', accuracy_score(testY, pred))
