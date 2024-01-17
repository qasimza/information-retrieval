# IR7A.py CS5154/6054 cheng 2022
# read lines from a text file as docs
# tokenize each as a bag of words
# make the inverted index
# randomly select a doc as the query with at least 5 words
# using the inverted index
# retrieve sets of docs containing each of the word in query
# update a dictionary called intersection with tf-idf
# call the top 10 tf-idf documents "relevant"
# then call documents with the top 10 Jaccard coefficient "retrieved"
# compute precision, recall, and F1
# they are all the same, why?
# Usage: python IR7A.py

import re
import numpy as np
import random
import math
from collections import Counter
from heapq import nlargest

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N = len(docs)
logN = math.log(N)

docLen = np.zeros(N, dtype=int)
counter = Counter()
invertedIndex = {}
for i in range(N):
    tokens = re.findall('\w+', docs[i])
    docLen[i] = len(tokens)
    counter.clear()
    counter.update(tokens)
    for t, tf in counter.items():
        if invertedIndex.get(t) == None:
            invertedIndex.update({t : {i : counter.get(t)}})
        else:
            invertedIndex.get(t).update({i : counter.get(t)})
       
query = random.randint(0, N)
print(query, docs[query])
tokens = re.findall('\w+', docs[query])
qlen = len(tokens)
counter.clear()
counter.update(tokens)
print(counter)

intersections = {}
jaccard = {}
for t, tf1 in counter.items():  
    idf = logN - math.log(len(invertedIndex.get(t)))
    for d, tf2 in invertedIndex.get(t).items():
        tfidf = tf1 * idf * tf2 * idf
        if intersections.get(d) == None:
            intersections.update({d : tfidf})
            jaccard.update({d : 1})
        else:
            x = intersections.get(d) + tfidf
            intersections.update({d : x})
            y = jaccard.get(d) + 1
            jaccard.update({d : y})

relevant = set(nlargest(10, intersections, key = intersections.get))
print('relevant', relevant)

for d, ab in jaccard.items():
    jaccard.update({d : ab / (qlen + docLen[d] - ab)})
retrieved = set(nlargest(10, jaccard, key = jaccard.get))
print('retrieved', retrieved)

# your code for precision, recall, and f1
"""
Problem Statement: IR7A.py uses an inverted index to find the top ten retrieval from both tf-idf (as in IR5B.py) 
and Jaccard coefficient between a random query and documents in a collection.  Let us call the top ten using tf-idf 
the relevant set and those using Jaccard coefficient the retrieved set.  Add code to compute precision, recall, 
and F1.  Run the resulting program at least five times and report the average precision. 
"""

precision = len(set(relevant) & set(retrieved))/len(retrieved)
recall = len(set(relevant) & set(retrieved))/len(relevant)
f1_score = 2 * precision * recall / (precision + recall)

print(f'Precision Score: {precision}')
print(f'Recall Score: {recall}')
print(f'F1 Score: {f1_score}')
