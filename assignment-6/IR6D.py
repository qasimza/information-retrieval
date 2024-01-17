# IR6D.py CS5154/6054 cheng 2022
# redo IR5A with TfidfVectorizer and cosine_similarity
# List the top five docs for the random query.
# Usage: python IR6D.py

import re
import random
import math
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


f = open("bible.txt", "r")
docs = f.readlines()
f.close()

tfidf = TfidfVectorizer(ngram_range=(1,2),max_df=0.4, min_df=3)
dt = tfidf.fit_transform(docs)
print(dt.shape)
print(dt.data.nbytes)

N = len(docs)
query = random.randint(0, N)
print(query, docs[query])

sim = cosine_similarity(dt[query], dt)
print(sim.shape)

res = np.flip(np.argsort(sim)[0, -5:])


for k in res:
    print(k, sim[0, k], docs[k])
