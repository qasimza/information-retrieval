# IR10A.py CS5154/6054 cheng 2022
# CountVectorizer (binary=True) makes a document-term matrix
# or the X vectors to be used by BernoulliNB along with an Y vector
# Test documents are transformed and the model is applied 
# the probability ranking score for relevance or being about China is reported.
# Usage: python IR10A.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB

docs = ["Chinese Beijing Chinese", "Chinese Chinese Shanghai", "Chinese Macao",
"Tokyo Japan Chinese"]
R = [1, 1, 1, 0]  # R=1: relevant, R=0: nonrelevant
tests = ["Chinese Chinese Chinese Tokyo Japan", "Shanghai Tokyo"]

cv = CountVectorizer(binary=True)
X = cv.fit_transform(docs).toarray()
voc = cv.get_feature_names()
doclen, voclen = X.shape
T = cv.transform(tests) # your code to transform (but not fit) tests with cv
print(X)
print(T)

model = BernoulliNB()
model.fit(X, R)
# your code to report the probability P(R=1|q, x), of course q is irrelevant now
print(f"Probability ranking score for relevance, P(R=1|q, x) is given in he following matrix\n"
      f"{model.predict_proba(T)}\n"
      f"where the corresponding classes are {model.classes_}")
