import nltk
nltk.download('brown')

from nltk.tokenize import word_tokenize
from nltk.util import ngrams
from nltk.corpus import brown
from nltk.chunk import RegexpParser

sentence = "The quick brown fox jumps over the lazy dog"
tokens = word_tokenize(sentence)
print(tokens)

pos_tags = nltk.pos_tag(tokens)
print("Part-of-speech Tagging:")
print(pos_tags)

text = brown.words(categories='news')[:1000]
bigrams = list(ngrams(text, 2))
freq_dist = nltk.FreqDist(bigrams)
print("\nN-gram Analysis(Bigrams with Smoothing):")
for bigram in bigrams:
    print(f"{bigram}:{freq_dist[bigram]}")

tagged_sentence = nltk.pos_tag(word_tokenize("The quick brown fox jumps over the lazy dog"))
grammar = r"NP: {<DT>?<JJ>*<NN>}"
cp = RegexpParser(grammar)
result = cp.parse(tagged_sentence)
print("\nChunking with Regular Expression and POS tags:")
print(result)