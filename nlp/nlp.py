# Natural Language Processing - General model in Python
# Example data: Predicting restaurant reviews

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
#nltk.download('stopwords')

# Importing the dataset
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter = '\t', quoting = 3)

# Cleaning the text
corpus = []
for i in range(0, 1000):
    # regex input - what types of characters we want to keep
    clean_review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])
    # change all characters to lower case
    clean_review = clean_review.lower()
    clean_review = clean_review.split()
    # object to use the stemming class
    ps = PorterStemmer()
    # remove words that are in stopwords list
    # stem the words - use only roots
    clean_review = [ps.stem(word) for word in clean_review if not word in set(stopwords.words('English'))]
    clean_review = ' '.join(clean_review)
    corpus.append(clean_review)
    
# Creating the Bag of Words model
# keep only 1500 most common words
cv = CountVectorizer(max_features = 1500)
# create sparse matrix - reviews & words in corpus
X = cv.fit_transform(corpus).toarray()
# dependent var vector - category +/- 
Y = dataset.iloc[:, 1].values

# Choose a classifier - Naive Bayes  

# Split data into training set and test set
from sklearn.cross_validation import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y,
                                                    test_size = 0.20, random_state = 0)

# Fitting Naive Bayes to training set 
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, Y_train)

# Predicting the test set results
Y_pred = classifier.predict(X_test)

# Confusion matrix to evaluate performance
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_test, Y_pred)
