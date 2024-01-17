# IR6C.py CS5154/6054 cheng 2022
# Usage: python IR6C.py

import re
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

f = open("gutprotocol.txt", "r", encoding="utf8")
docs = f.readlines()
f.close()
tv = TfidfVectorizer(max_df=0.4, min_df=3)
tv.fit(docs)
print(tv.get_feature_names())
vectors = tv.transform(docs)
print(vectors)
