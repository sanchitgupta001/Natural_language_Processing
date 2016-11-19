from nltk.tokenize import sent_tokenize, word_tokenize

# nltk.download() # Run this command only once when you start 

# Tokenizing - word tokenizers, sentence tokenizers
# Lexicon and corpora
# corpora : body of text. Ex : English Language
# lexicon : words and their meaning

example_text = "Hello Mr. Smith, how are you doing today? The weather is great and Python is awesome. I love programming in Python. Python is a very friendly language."

# Print tokenized words and sentences
#print(sent_tokenize(example_text))
#print(word_tokenize(example_text))

for i in sent_tokenize(example_text):
    print(i)

for i in word_tokenize(example_text):
    print(i)