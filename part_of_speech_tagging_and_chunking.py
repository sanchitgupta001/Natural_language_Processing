import nltk
from nltk.corpus import state_union
from nltk.tokenize import PunktSentenceTokenizer # Unsupervised machine learning tokenizer

train_text = state_union.raw('2005-GWBush.txt')
sample_text = state_union.raw('2006-GWBush.txt')

custom_sent_tokenizer = PunktSentenceTokenizer(sample_text)

tokenized = custom_sent_tokenizer.tokenize(sample_text)

#==============================================================================
# Modifiers:
# 
# {1,3} = for digits, u expect 1-3 counts of digits, or "places"
# + = match 1 or more
# ? = match 0 or 1 repetitions.
# * = match 0 or MORE repetitions
# $ = matches at the end of string
# ^ = matches start of a string
# | = matches either/or. Example x|y = will match either x or y
# [] = range, or "variance"
# {x} = expect to see this amount of the preceding code.
# {x,y} = expect to see this x-y amounts of the precedng code
#==============================================================================

# Parts of speech, Chunking
def process_content():
    try:
        for i in tokenized:
            words = nltk.word_tokenize(i)
            #The process of classifying words into their parts of speech and labeling them accordingly is known as part-of-speech tagging, POS-tagging, or simply tagging. 
            tagged = nltk.pos_tag(words)
            #print(tagged)
            
            chunkGram = r"""Chunk: {<RB.?>*<VB.?>*<NNP>+<NN>?}""" # r at the starting denotes Regular expression
            # Chunking is a term referring to the process of taking individual pieces of information (chunks) and grouping them into larger units. 
            # By grouping each piece into a large whole, you can improve the amount of information you can remember.
            
            chunkParser = nltk.RegexpParser(chunkGram)
            chunked = chunkParser.parse(tagged)
            
            chunked.draw()
    
    except Exception as e:
        print(str(e))

process_content()                 