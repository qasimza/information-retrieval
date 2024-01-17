# IR12A.py CS5154/6054 cheng 2022
# This is the example from sklearn.metrics.dcg_score
# Usage: IR12A.py

import numpy as np
from sklearn.metrics import dcg_score
# we have groud-truth relevance of some answers to a query:
true_relevance = np.asarray([[10, 0, 0, 1, 5]])
# we predict scores for the answers
scores = np.asarray([[.1, .2, .3, 4, 70]])
print(dcg_score(true_relevance, scores))
# we can set k to truncate the sum; only top k answers contribute
print(dcg_score(true_relevance, scores, k=2))
# now we have some ties in our prediction
scores = np.asarray([[1, 0, 0, 0, 1]])
# by default ties are averaged, so here we get the average true
# relevance of our top predictions: (10 + 5) / 2 = 7.5
print(dcg_score(true_relevance, scores, k=1))
# we can choose to ignore ties for faster results, but only
# if we know there aren't ties in our scores, otherwise we get
# wrong results:
print(dcg_score(true_relevance, scores, k=1, ignore_ties=True))
