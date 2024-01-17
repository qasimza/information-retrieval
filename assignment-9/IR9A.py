# IR9A.py CS5154/6054 cheng 2022
# modified from chapter 5 of Blueprints for text analytics using Python 
# CountVectorizer (binary=True) is used to make the term-document matrix A
# cosine similarity is used to find normalized co-occurrence of pairs of terms
# argsort is used to find the pairs with the top scores
# documents where they co-occur are printed out
# Usage: python IR9A.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open("bible.txt", "r")
docs = f.readlines()
f.close()

cv = CountVectorizer(binary=True, max_df=0.04, min_df=8)
A = cv.fit_transform(docs).toarray().T
voc = cv.get_feature_names()
voclen, doclen = A.shape
r = cosine_similarity(A, A)
np.fill_diagonal(r, 0)

for index in np.argsort(r.flatten())[::-1][0:10]:
    a = int(index / voclen)
    b = index % voclen
    if a > b:
        print(r[a][b], voc[a], voc[b])
'''
        for i in range(doclen):
            if A[a][i] > 0 and A[b][i] > 0:
                print(i, docs[i])
'''
