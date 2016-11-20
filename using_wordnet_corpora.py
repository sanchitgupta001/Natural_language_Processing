from nltk.corpus import wordnet

## Synonyms Set
#synonyms = wordnet.synsets("program")
#
## A lemma (in linguistics) is the canonical form, or morphological form, of a word.
## Just the word
#print(synonyms[0].lemmas()[0].name())
#
## definition
#print(synonyms[0].definition())
#
## Examples
#print(synonyms[0].examples())

synonyms = []
antonyms = []

for syn in wordnet.synsets("good"):
    for l in syn.lemmas():
        synonyms.append(l.name())
        if l.antonyms():
            antonyms.append(l.antonyms()[0].name())
            
            
print(synonyms)
print(antonyms) 

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('boat.n.01')

# Find sematic similarity
print(w1.wup_similarity(w2))

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('car.n.01')

# Find sematic similarity
print(w1.wup_similarity(w2)) 

w1 = wordnet.synset('ship.n.01')
w2 = wordnet.synset('goat.n.01')

# Find sematic similarity
print(w1.wup_similarity(w2))           