# IR1B.py CS5154/6054 cheng 2022
# read lines from a text file
# turn each line into a list of tokens
# make the inverted index
# retrieve documents using boolean queries
# Usage: python IR1B.py

f = open("bible.txt", "r")
docs = f.readlines()
f.close()
invertedIndex = {}
for i in range(len(docs)):
    for s in docs[i].split():
        if invertedIndex.get(s) == None:
            invertedIndex.update({s : {i}})
        else:
            invertedIndex.get(s).add(i)
       
word1 = 'punishment'
word2 = 'transgressions'
for j in invertedIndex.get(word1) & invertedIndex.get(word2):
    print(j, docs[j])

