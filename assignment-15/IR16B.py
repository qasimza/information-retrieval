# IR16B.py CS5154/6054 cheng 2022
# three classes
# Usage: python IR16B.py

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import BernoulliNB
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
N =len(docs)
mid = 1100 + (N - 3 * 1100) // 2
trainX = np.concatenate([docs[0:1000], docs[mid:mid+1000], docs[N-1000:N]])
y = np.concatenate([np.zeros(1000, dtype=np.int16), np.ones(1000, dtype=np.int16), np.full(1000, 2, dtype=np.int16)])
testX = np.concatenate([docs[1000:1100], docs[mid+1000:mid+1100], docs[N-1100:N-1000]])
testY = np.concatenate([np.zeros(100, dtype=np.int16), np.ones(100, dtype=np.int16), np.full(100, 2, dtype=np.int16)])

# documents as binary vectors
cv = CountVectorizer(binary=True, max_df=0.4, min_df=4)
X0 = cv.fit_transform(trainX).toarray()
T0 = cv.transform(testX).toarray()

model = BernoulliNB()
model.fit(X0, y)
pred = model.predict(T0) 
print(pred)
A0 = accuracy_score(testY, pred)
print ('BernoulliNB -', A0)

cm = confusion_matrix(testY, pred)
print(cm)
disp = ConfusionMatrixDisplay(cm, display_labels=model.classes_)
disp.plot()
plt.show()
