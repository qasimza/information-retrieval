# IR2A.py CS5154/6054 cheng 2022
# read lines from a text file
# tokenize and count words
# use WordCloud to show words with top frequencies
# Usage: python IR2A.py bible.txt

import re
import sys
from collections import Counter
from wordcloud import WordCloud
from matplotlib import pyplot as plt

f = open(sys.argv[1], 'r')
       
counter = Counter()
for t in f:
    counter.update(re.findall('\w+', t))

wc = WordCloud()
wc.generate_from_frequencies(counter)
plt.imshow(wc)
plt.axis("off")
plt.show()
