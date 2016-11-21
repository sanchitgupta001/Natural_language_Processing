import nltk
import random
from nltk.corpus import movie_reviews
import pickle
from nltk.tokenize import word_tokenize

from nltk.classify.scikitlearn import SklearnClassifier
from sklearn.naive_bayes import MultinomialNB, GaussianNB, BernoulliNB
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.svm import SVC, LinearSVC, NuSVC

from nltk.classify import ClassifierI
from statistics import mode

# VOTED CLASSIFIER
class VoteClassifier(ClassifierI):
    def __init__(self, *classifiers):
        self._classifiers = classifiers
    
    def classify(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
        return mode(votes)
        
    def confidence(self, features):
        votes = []
        for c in self._classifiers:
            v = c.classify(features)
            votes.append(v)
            
        choice_votes = votes.count(mode(votes))
        conf = choice_votes/len(votes)
        return conf

short_pos = open('positive.txt','r').read()
short_neg = open('negative.txt','r').read()

documents = []

# Load positive dataset in documents
for r in short_pos.split('\n'):
    documents.append((r, 'pos'))

# Load negative dataset in documents    
for r in short_neg.split('\n'):
    documents.append((r, 'neg'))


all_words = []

short_pos_words = word_tokenize(short_pos)    
short_neg_words = word_tokenize(short_neg)    

for w in short_pos_words:
    all_words.append(w.lower())
    
for w in short_neg_words:
    all_words.append(w.lower())    
    
# Find the frequency distribution for all_words    
all_words = nltk.FreqDist(all_words)

# Print 15 most_common used words
#print(all_words.most_common(15))   

# Print number of times the word 'stupid' appears in all reviews
#print(all_words['stupid']) 

word_features = list(all_words.keys())[:5000]

def find_features(document):
    words = word_tokenize(document)
    features = {}
    for w in word_features:
        features[w] = (w in words)
        
    return features

#print((find_features(movie_reviews.words('neg/cv000_29416.txt'))))

featuresets = [(find_features(rev), category) for (rev, category) in documents]    

random.shuffle(featuresets)

training_set = featuresets[:5000]
testing_set = featuresets[5000:6000]

classifier = nltk.NaiveBayesClassifier.train(training_set)

#classifier_f = open('naiveBayes.pickle','rb')
#classifier = pickle.load(classifier_f)
#classifier_f.close()

print("Original Naive Bayes Algo accuracy percent: ", (nltk.classify.accuracy(classifier, testing_set))*100)
#classifier.show_most_informative_features(15)

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
#SVC_classifier = SklearnClassifier(SVC())
#SVC_classifier.train(training_set)
#print("SVC_classifier accuracy percent: ", (nltk.classify.accuracy(SVC_classifier, testing_set))*100)

# LinearSVC Classifier
LinearSVC_classifier = SklearnClassifier(LinearSVC())
LinearSVC_classifier.train(training_set)
print("LinearSVC_classifier accuracy percent: ", (nltk.classify.accuracy(LinearSVC_classifier, testing_set))*100)

# NuSVC Classifier
NuSVC_classifier = SklearnClassifier(NuSVC())
NuSVC_classifier.train(training_set)
print("NuSVC_classifier accuracy percent: ", (nltk.classify.accuracy(NuSVC_classifier, testing_set))*100)

# New Voted Classifier
voted_classifier = VoteClassifier(classifier, MNB_classifier, BNB_classifier, LogisticRegression_classifier, SGDC_classifier, LinearSVC_classifier, NuSVC_classifier)

print("voted_classifier accuracy percent: ", (nltk.classify.accuracy(voted_classifier, testing_set))*100)

print('Classification : ', voted_classifier.classify(testing_set[0][0])," Confidence % : ", voted_classifier.confidence(testing_set[0][0])*100)
print('Classification : ', voted_classifier.classify(testing_set[1][0])," Confidence % : ", voted_classifier.confidence(testing_set[1][0])*100)
print('Classification : ', voted_classifier.classify(testing_set[2][0])," Confidence % : ", voted_classifier.confidence(testing_set[2][0])*100)
print('Classification : ', voted_classifier.classify(testing_set[3][0])," Confidence % : ", voted_classifier.confidence(testing_set[3][0])*100)
print('Classification : ', voted_classifier.classify(testing_set[4][0])," Confidence % : ", voted_classifier.confidence(testing_set[4][0])*100)
print('Classification : ', voted_classifier.classify(testing_set[5][0])," Confidence % : ", voted_classifier.confidence(testing_set[5][0])*100)