import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer # Unsupervised machine learning tokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

# Parts of speech
def process_content():
    try:
        for i in tokenized[:5]:
            words = nltk.word_tokenize(i)
            #The process of classifying words into their parts of speech and labeling them accordingly is known as part-of-speech tagging, POS-tagging, or simply tagging. 
            tagged = nltk.pos_tag(words)
            
            # Named entity recognition
            namedEnt = nltk.ne_chunk(tagged)
            
            namedEnt.draw()
            
            
    
    except Exception as e:
        print(str(e))

process_content()                 