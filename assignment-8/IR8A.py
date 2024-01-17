# IR8A.py CS5154/6054 cheng 2022
# TfidfVectorizer and CountVectorizer (binary=True) are used 
# a random doc is the query and the top 50 cosine similarity
# in Tfidf are considered relevent
# CountVectors are ranked using cosine similarity
# precision and recall at each retrieval level are computed
# and the precision-recall graph (Fig 8.2 iir) is plotted
# Usage: python IR8A.py

import re
import numpy as np
import random
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from matplotlib import pyplot as plt
relevant = 50

f = open("bible.txt", "r")
docs = f.readlines()
f.close()

tfidf = TfidfVectorizer(max_df=0.4, min_df=2)
dt = tfidf.fit_transform(docs)
N = len(docs)
query = random.randint(0, N)
print(query, docs[query])

sim = cosine_similarity(dt[query], dt)
toptfidf = set()
for index in np.argsort(sim)[0][::-1][0:relevant]:
    toptfidf.add(index)

print(toptfidf)

cv = CountVectorizer(binary=True, max_df=0.4, min_df=2)
dt2 = cv.fit_transform(docs)
sim2 = cosine_similarity(dt2[query], dt2)
sorted = np.argsort(sim2)[0][::-1]
precision = np.zeros(N)
recall = np.zeros(N)
m = 0
for i in range(N):
    if sorted[i] in toptfidf:
        m = m + 1
        # tp = m, fn = relevant - m, fp = i + 1 - m, tn = N - tp - fn - fp
    precision[i] = m/(i+1)  # Precision = tp/(tp+fp)
    recall[i] = m/relevant  # Recall = tp/(tp+fn)

plt.scatter(recall, precision)
plt.show()

"""
Once we have the precision and recall arrays, it should not be too difficult to go through them to find MAP 
(average of precisions every time recall does not decrease) and R-precision (when you have retrieved 50 documents).  
Add code so you can report these two values.
"""

average_precision = (1/relevant) * (1 if recall[0] > 0 else 0) \
                    + sum([precision[i] for i in range(1, N) if recall[i] > recall[i-1]])
r_precision = precision[relevant]

print(f"Mean Average Precision, mAP: {average_precision}")
print(f"R-Precision: {r_precision}")

"""
Add code on slide 9/15/23 to the program, so you can generate a eleven-point interpolated precision-recall graph 
(iir fig. 8.3) for a single query.
"""

eleven_recalls = np.zeros(11)
interpolated = np.zeros(11)
n = 0
for i in range(N):
    if n <= 10 and recall[i] * 10 >= n:
        interpolated[n] = max(precision[i:])
        eleven_recalls[n] = recall[i]
        print(n, precision[i], interpolated[n])
        n = n + 1
    if n > 10:
        break
plt.scatter(eleven_recalls, interpolated)
plt.show()

"""
Replace precision in IR8A by 1-specificity (rocx) and produce the ROC curve (iir fig. 8.4) for a query.
"""
rocx = np.zeros(N)
m = 0
for i in range(N):
    if sorted[i] in toptfidf:
        m = m + 1
        rocx[i] = (i + 1 - m)/(N - relevant)
plt.scatter(rocx, recall)
plt.show()