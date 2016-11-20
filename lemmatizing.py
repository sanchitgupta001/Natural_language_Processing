from nltk.stem import WordNetLemmatizer

lemmatizer = WordNetLemmatizer()

print(lemmatizer.lemmatize('cats'))
print(lemmatizer.lemmatize('loci'))
print(lemmatizer.lemmatize('geese'))
print(lemmatizer.lemmatize('rocks'))
print(lemmatizer.lemmatize('python'))

print(lemmatizer.lemmatize('better',pos='a')) # As an adjective
print(lemmatizer.lemmatize('run',pos='a')) # As an adjective
print(lemmatizer.lemmatize('run')) # As a noun
print(lemmatizer.lemmatize('best',pos='a')) # As an adjective
