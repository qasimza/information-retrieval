# IR3A.py CS5154/6054 cheng 2022
# read lines from a text file as documents 
# tokenize each into a set
# randomly name one doc as query
# compute Jaccard coefficient between query and all docs
# print docs with Jaccard coefficient > 0.1
# Usage: python IR3A.py

import re
import random

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N = len(docs)
sets = list(map(lambda s: set(re.findall('\w+', s)), docs))
query = random.randint(0, N)
print(query)
print(docs[query])
A = sets[query]
for i in range(N):
    B = sets[i]
    C = A & B
    if len(C) == 0:
        continue
    D = A | B
    jaccard = len(C) / len(D)
    if jaccard > 0.1:
        print(i, docs[i], jaccard, C)


