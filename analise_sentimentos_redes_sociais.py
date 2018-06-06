# Natural Language Processing

# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
dataset = pd.read_csv('twitters_classificados.csv', delimiter = ';', quoting = 3)

# Cleaning the texts
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
for i in range(0, 1418):
    review = re.sub('[^a-zA-Z]', ' ', dataset['Text'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('portuguese'))]
    review = ' '.join(review)
    corpus.append(review)
    

# Creating the Bag of Words model
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer()
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, 1].values

# Splitting the dataset into the Training set and Test set
from sklearn.cross_validation import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.30, random_state = 35)

# Fitting Naive Bayes to the Training set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)

# Predicting the Test set results
y_pred = classifier.predict(X_test)

# Accuracy
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred) * 100

# Making the Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

# Frequency of words
from nltk.probability import FreqDist
import string  
words = dataset.iloc[:, 0].values
fdist = {}
a = ''.join(words)
b = ''.join(['' if ch in string.punctuation else ch for ch in a])
tokens = nltk.tokenize.word_tokenize(b)  
fdist = FreqDist(tokens)             
mostWords = fdist.most_common(100)


#tokens = nltk.word_tokenize(words)
#fdist=FreqDist(tokens)
#for sentence in nltk.tokenize.sent_tokenize(tokenized_sents):
#    for i in nltk.tokenize.word_tokenize(sentence):
#         fdist[i] += 1   
#for i in tokenized_sents:
#    fdist[i].append(i)
# tokenized_sents = [nltk.word_tokenize(i) for i in words]