# IR12C.py CS5154/6054 cheng 2022
# An example from sklearn User Guide 3.3.3.2 Label ranking average precision
# Usage: python IR12C.py

import numpy as np
from sklearn.metrics import label_ranking_average_precision_score
y_true = np.array([[1, 0, 0], [0, 0, 1]])
y_score = np.array([[0.75, 0.5, 1], [1, 0.2, 0.1]])
print(label_ranking_average_precision_score(y_true, y_score))