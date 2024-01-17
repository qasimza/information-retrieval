# IR18A.py CS5154/6054 cheng 2022
# k-means by TrainRocchio and ApplyRocchio
# Usage: python IR18A.py

import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestCentroid
from matplotlib import pyplot as plt
from sklearn.metrics import accuracy_score

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N = len(docs)
K = 8

cv = TfidfVectorizer(max_df=0.4, min_df=4)
X = cv.fit_transform(docs)
seeds = random.sample(range(N), K)
centroids = X[seeds]
classes = list(range(K))
init = NearestCentroid()
init.fit(centroids, classes)
y = init.predict(X)
#plt.hist(y)
#plt.show()

model = NearestCentroid()
for iter in range(5):
    model.fit(X, y)
    pred = model.predict(X)
    print(accuracy_score(y, pred))
    y = pred
    
RSS = 0
for i in range(N):
    d = X[i] - model.centroids_[y[i]]
    RSS += np.dot(d, d.T).item()
print(RSS)
