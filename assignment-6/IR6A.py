# IR6A.py CS5154/6054 cheng 2022
# Usage: python IR6A.py

import re
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics.pairwise import cosine_similarity

f = open("iirexercise1-2.txt", "r")
docs = f.readlines()
f.close()
tokens = list(map(lambda s: re.findall('\w+', s), docs))
lb = MultiLabelBinarizer()
lb.fit(tokens)
print(lb.classes_)
vectors = lb.transform(tokens)
print(vectors)

print(np.dot(vectors[0], vectors[1]))
print(cosine_similarity(vectors))