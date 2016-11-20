from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize

example_sentence = 'This is an example showing of stop word filtration.'

# Stopwords are the words that are automatically omitted from a computer-generated concordance or index
# Not useful for data analysis
stop_words = set(stopwords.words('english'))

words = word_tokenize(example_sentence)

filtered_sentence = []

for w in words:
    if w not in stop_words:
        filtered_sentence.append(w)
   
# One line format
#filtered_sentence = [w for w in words if w not in stop_words]        
print(filtered_sentence)        
