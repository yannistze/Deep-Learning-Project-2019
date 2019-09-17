import nltk

# check if the corpus is downloaded already
try:
    from nltk.corpus import brown
    words = brown.words()
    print('NLTK Corpus already downloaded')

except:
    print('Downloading NLTK corpus')
    nltk.download()
