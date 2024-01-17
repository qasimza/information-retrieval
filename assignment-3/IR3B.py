# IR3B.py CS5154/6054 cheng 2022
# read lines from a text file as docs
# tokenize each as a set of words
# make the inverted index
# name those words with postings list longer than 1000 stopwords
# remove stopwords from the sets representing docs
# randomly select a doc as the query
# compute Jaccard coefficients by set intersection and union
# list docs with Jaccard coefficient > 0.2
# Usage: python IR3B.py

import re
import random

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
invertedIndex = {}
for i in range(len(docs)):
    for s in set(re.findall('\w+', docs[i])):
        if invertedIndex.get(s) == None:
            invertedIndex.update({s : {i}})
        else:
            invertedIndex.get(s).add(i)
       
stopwords = set()
for k, v in invertedIndex.items():
    if len(v) > 1000:
        stopwords.add(k)

N = len(docs)
sets = list(map(lambda s: set(re.findall('\w+', s)) - stopwords, docs))
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
    if jaccard > 0.2:
        print(i, docs[i], jaccard, C)
