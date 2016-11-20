import nltk
from nltk.corpus import gutenberg
from nltk.tokenize import sent_tokenize

# print(nltk.__file__) # To find the location of __init__.py file

sample = gutenberg.raw('bible-kjv.txt')

token = sent_tokenize(sample)

print(token[:5])