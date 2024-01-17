# IR19A.py CS5154/6054 cheng 2022
# twice k-means
# confusion matrix
# NMI
# Usage: python IR19A.py

import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, normalized_mutual_info_score
import matplotlib.pyplot as plt

f = open("gutprotocol.txt", "r", encoding="utf8")
docs = f.readlines()
f.close()
N = len(docs)

cv = TfidfVectorizer(max_df=0.4, min_df=4)
X = cv.fit_transform(docs)

model = KMeans(n_init=1, max_iter=5)
model.fit_predict(X)
y1 = model.labels_
print(y1)

model.fit_predict(X)
y2 = model.labels_
print(y2)

cm = confusion_matrix(y1, y2)
print(cm)
disp = ConfusionMatrixDisplay(cm)
disp.plot()
plt.show()

print(normalized_mutual_info_score(y1, y2))

inverted = {}
for i in range(N):
    if not y1[i] in inverted:
        s = set()
        s.add(i)
        inverted.update({y1[i]: s})
    else:
        inverted.get(y1[i]).add(i)

for x, y in inverted.items():
    print(x, y)
