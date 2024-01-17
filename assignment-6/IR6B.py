# IR6B.py CS5154/6054 cheng 2022
# Usage: python IR6B.py

import re
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open("gutprotocol.txt", "r", encoding="utf8")
docs = f.readlines()
f.close()
cv = CountVectorizer()
cv.fit(docs)
print(cv.get_feature_names())
vectors = cv.transform(docs)
print(vectors)
