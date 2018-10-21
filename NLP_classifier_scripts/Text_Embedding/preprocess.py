import numpy as np
import math
import re
from os import listdir
from tqdm import tqdm

from os.path import isfile, join
np.set_printoptions(threshold=np.inf)

#Functions
def cleanSentences(string):
    string = string.lower().replace("<br />", " ")
    return re.sub(strip_special_chars, "", string.lower())


wordsList = open("motAwk.txt").readlines()
wordsList = [s.strip('\n') for s in wordsList]
print("wordlist loaded")

#data from http://ai.stanford.edu/%7Eamaas/data/sentiment/aclImdb_v1.tar.gz
positiveFiles = ['aclImdb/train/pos/' + f for f in listdir('aclImdb/train/pos/') if isfile(join('aclImdb/train/pos/', f))]
negativeFiles = ['aclImdb/train/neg/' + f for f in listdir('aclImdb/train/neg/') if isfile(join('aclImdb/train/neg/', f))]

numFiles=len(positiveFiles)+len(negativeFiles)
print("NumFiles ",numFiles)

maxSeqLength = 250 # Arbitrary number
ids = np.zeros((numFiles, maxSeqLength), dtype='int32')
fileCounter = 0
unkown = len(wordsList)+2



for pf in tqdm(positiveFiles):
    indexCounter=0

    with open(pf, "r") as f:
        indexCounter = 0
        line=f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = unkown #Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter == maxSeqLength-1:
                ids[fileCounter][indexCounter] = 1
                break
        fileCounter = fileCounter + 1
        if fileCounter % 100 == 0:
            print(fileCounter)


for nf in tqdm(negativeFiles):
    indexCounter=0

    with open(nf, "r") as f:
        indexCounter = 0
        line=f.readline()
        cleanedLine = cleanSentences(line)
        split = cleanedLine.split()
        for word in split:
            try:
                ids[fileCounter][indexCounter] = wordsList.index(word)
            except ValueError:
                ids[fileCounter][indexCounter] = unkown #Vector for unkown words
            indexCounter = indexCounter + 1
            if indexCounter == maxSeqLength-1:
                ids[fileCounter][indexCounter] = 0
                break
        fileCounter = fileCounter + 1
        if fileCounter % 100 == 0:
            print(fileCounter)

#Pass into embedding function and see if it evaluates.

np.random.shuffle(ids)
print("Ids",ids[0:10,:])

np.save('idsMatrix', ids)
