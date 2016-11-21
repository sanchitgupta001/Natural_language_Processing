import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

# Making tuples
documents = [(list(movie_reviews.words(fileId)), category)
             for category in movie_reviews.categories()
             for fileId in movie_reviews.fileids(category)]

random.shuffle(documents)

# print(documents[1])

all_words = []
for w in movie_reviews.words():
    all_words.append(w.lower())
    
# Find the frequency distribution for all_words    
all_words = nltk.FreqDist(all_words)

# Print 15 most_common used words
#print(all_words.most_common(15))   

# Print number of times the word 'stupid' appears in all reviews
#print(all_words['stupid']) 

word_features = list(all_words.keys())[:3000]

def find_features(document):
    words = set(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]    

training_set = featuresets[:1900]
testing_set = featuresets[1900:]

#classifier = nltk.NaiveBayesClassifier.train(training_set)

classifier_f = open('naiveBayes.pickle','rb')
classifier = pickle.load(classifier_f)
classifier_f.close()

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
classifier.show_most_informative_features(15)

#save_classifier = open('naiveBayes.pickle',"wb")
#pickle.dump(classifier, save_classifier)
#save_classifier.close()

# MNB Classifier
MNB_classifier = SklearnClassifier(MultinomialNB())
MNB_classifier.train(training_set)
print("MNB_classifier accuracy percent: ", (nltk.classify.accuracy(MNB_classifier, testing_set))*100)

# GaussianNB Classifier
#GNB_classifier = SklearnClassifier(GaussianNB())
#GNB_classifier.train(training_set)
#print("GNB_classifier accuracy percent: ", (nltk.classify.accuracy(GNB_classifier, testing_set))*100)

# BernoulliNB Classifier
BNB_classifier = SklearnClassifier(BernoulliNB())
BNB_classifier.train(training_set)
print("BNB_classifier accuracy percent: ", (nltk.classify.accuracy(BNB_classifier, testing_set))*100)

# Logistic Regression Classifier
LogisticRegression_classifier = SklearnClassifier(LogisticRegression())
LogisticRegression_classifier.train(training_set)
print("LogisticRegression_classifier accuracy percent: ", (nltk.classify.accuracy(LogisticRegression_classifier, testing_set))*100)

# SGDC Classifier
SGDC_classifier = SklearnClassifier(SGDClassifier())
SGDC_classifier.train(training_set)
print("SGDC_classifier accuracy percent: ", (nltk.classify.accuracy(SGDC_classifier, testing_set))*100)

# SVC Classifier
SVC_classifier = SklearnClassifier(SVC())
SVC_classifier.train(training_set)
print("SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LinearSVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC Classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)