# IR1A.py CS5154/6054 cheng 2022
# read lines from a text file
# retrieve documents for boolean queries
# Usage: python IR1A.py

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
       
word1 = 'punishment'
word2 = 'transgressions'
for i in range(len(docs)):
    if word1 in docs[i] and word2 in docs[i]:
        print(i, docs[i])
