import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import json
import string
import ast
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem import SnowballStemmer
from stemming.porter2 import stem
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, confusion_matrix
from sklearn.feature_extraction import DictVectorizer
from collections import Counter, defaultdict

# Azlyrics function; used for fetching lyrics data through API

agent = 'Mozilla/5.0 (Windows NT 6.0; WOW64; rv:24.0) \
        Gecko/20100101 Firefox/24.0'
headers = {'User-Agent': agent}
base = "https://www.azlyrics.com/"


def songs(artist):
    artist = artist.lower().replace(" ", "")
    first_char = artist[0]
    url = base+first_char+"/"+artist+".html"
    req = requests.get(url, headers=headers)

    artist = {
        'artist': artist,
        'albums': {}
        }

    soup = BeautifulSoup(req.content, 'html.parser')

    all_albums = soup.find('div', id='listAlbum')
    first_album = all_albums.find('div', class_='album')
    album_name = first_album.b.text
    songs = []

    for tag in first_album.find_next_siblings(['a', 'div']):
        if tag.name == 'div':
            artist['albums'][album_name] = songs
            songs = []
            if tag.b is None:
                pass
            elif tag.b:
                album_name = tag.b.text

        else:
            if tag.text is "":
                pass
            elif tag.text:
                songs.append(tag.text)

    artist['albums'][album_name] = songs

    return (json.dumps(artist))


def lyrics(artist, song):
    artist = artist.lower().replace(" ", "")
    song = song.lower().replace(" ", "")
    url = base+"lyrics/"+artist+"/"+song+".html"

    req = requests.get(url, headers=headers)
    soup = BeautifulSoup(req.content, "html.parser")
    lyrics = soup.find_all("div", attrs={"class": None, "id": None})
    if not lyrics:
        return {'Error': 'Unable to find '+song+' by '+artist}
    elif lyrics:
        lyrics = [x.getText() for x in lyrics]
        return lyrics


def artists(letter):
    if letter.isalpha() and len(letter) is 1:
        letter = letter.lower()
        url = base+letter+".html"
        req = requests.get(url, headers=headers)
        soup = BeautifulSoup(req.content, "html.parser")
        data = []

        for div in soup.find_all("div", {"class": "container main-page"}):
            links = div.findAll('a')
            for a in links:
                data.append(a.text.strip())
        return json.dumps(data)
    else:
        raise Exception("Unexpected Input")


# NLP functions

# Find bracketed song structure words; such words as "[Chorus:]", "[Verse 1:]", etc...
# Find all the patterns "[*****]"
def bracketed_song_structure_word_search(sentence):
    import re
    match = re.findall(r'\[(.+)\]', sentence)
    if match:
        return match
    else:
        return None


# Find non-bracketed song structure words; such words as "Chorus:", "Verse 1:", etc...
# Find all the patterns "******:"
def song_structure_word_search(sentence):
    import re
    match = re.findall(r'(.+):', sentence)
    if match:
        return match
    else:
        return None


# Find bracketed chorus words
# Find all the patterns "(*****)"
def bracketed_chorus_word_search(sentence):
    import re
    match = re.findall(r'\((.+)\)', sentence)
    if match:
        return match
    else:
        return None


# Process lyrics
def lyrics_processor(sentence):
    if str(sentence) == 'nan':
        return None
    else:
        # Remove tab if exists
        sentence = sentence.replace('\t', ' ')
        # Trim sentence
        sentence = sentence


        # Remove bracketed song structure words; such words as "[Chorus:]", "[Verse 1:]", etc...
        if bracketed_song_structure_word_search(sentence):
            for s in bracketed_song_structure_word_search(sentence):
                s = '[' + s + ']'
                sentence = sentence.replace(s, '').strip()

        # Remove non-bracketed song structure words; such words as "Chorus:", "Verse 1:", etc...
        if song_structure_word_search(sentence):
            for s in song_structure_word_search(sentence):
                s = s + ':'
                sentence = sentence.replace(s, '').strip()

        # Remove bracketed chorus words
        if bracketed_chorus_word_search(sentence):
            for s in bracketed_chorus_word_search(sentence):
                s = '(' + s + ')'
                sentence = sentence.replace(s, '').strip()

        # remove "\n\n"
        sentence = sentence.replace('\n\n', '.')
        # remove "\n"
        sentence = sentence.replace('\n', ' ')
        # remove "\r"
        sentence = sentence.replace('\r', ' ')

        return sentence


# Function to create dictionary of unique genres and corresponding one hot vectors from dataset
def create_artist_dict(df):
    artists = df['artist'].unique()
    artist_dict = dict()
    for i in range(len(artists)):
        artist = artists[i]
        one_hot_vector = np.zeros(len(artists), dtype=int)
        one_hot_vector[i] = 1
        one_hot_vector = list(map(lambda x: str(x), one_hot_vector))
        one_hot_vector = ('').join(one_hot_vector)
        class_label = i
        artist_dict.update({str(artist): [one_hot_vector, class_label]})
    return artist_dict


# Function to obtain one hot vector for genre
def get_one_hot_vector_artist(artist):
    return artist_dict[artist][0]


# Function to obtain integer class label for genre
def get_class_label(artist):
    return artist_dict[artist][1]


def tokenize(s):
    s = "".join(" " if x in string.punctuation else x for x in s.lower())
    return s.split()

# Lemmatize and Remove Stop words
lemmatizer = WordNetLemmatizer()
ps = PorterStemmer()
ls = LancasterStemmer()
ss = SnowballStemmer("english")

stop_words = stopwords.words('english')

for s in string.punctuation:
    stop_words.append(s)

for s in '1234567890':
    stop_words.append(s)

def lemmatize(sentence):
    return [lemmatizer.lemmatize(word) for word in sentence]

def stemming(sentence):
    #return [ps.stem(word) for word in sentence]
    #return [ls.stem(word) for word in sentence]
    return [ss.stem(word) for word in sentence]
    #return [stem(word) for word in sentence]

def remove_stopwords(sentence):
    return [word for word in sentence if word not in stop_words]

def remove_low_freq_words(sentence):
    return [word for word in sentence if word not in low_freq_words]


# Create Metallica / Nirvana Datasets
nirvana_df = pd.read_csv('nirvana_df.tsv', sep='\t')
nirvana_df = nirvana_df[['artist', 'song', 'lyrics']]
metallica_df = pd.read_csv('metallica_df.tsv', sep='\t')
metallica_df = metallica_df[['artist', 'song', 'lyrics']]
frames = [nirvana_df, metallica_df]
lyrics_metallica_nirvana = pd.concat(frames)
lyrics_metallica_nirvana["lyrics"] = lyrics_metallica_nirvana["lyrics"].apply(lyrics_processor)


# Add NLP processing
# Process lyrics string
lyrics_metallica_nirvana["lyrics"] = lyrics_metallica_nirvana["lyrics"].apply(str.lower)
# lyrics_metallica_nirvana = lyrics_metallica_nirvana[lyrics_metallica_nirvana["lyrics"].str.len() >= 200]
lyrics_metallica_nirvana["lyrics"] = lyrics_metallica_nirvana['lyrics'].apply(tokenize)
#lyrics_metallica_nirvana["lyrics"] = lyrics_metallica_nirvana['lyrics'].apply(stemming)
lyrics_metallica_nirvana["lyrics"] = lyrics_metallica_nirvana['lyrics'].apply(lemmatize)
lyrics_metallica_nirvana["lyrics"] = lyrics_metallica_nirvana['lyrics'].apply(remove_stopwords)


# Process low freq words

vocab_dict = dict()

for lyrics in lyrics_metallica_nirvana['lyrics']:
    for k,v in Counter(lyrics).items():
        if k in vocab_dict.keys():
            vocab_dict[k] += v
        else:
            vocab_dict.update({k:v})

low_freq_words = [k for k, v in vocab_dict.items() if v < 8]

# Nirvana Vocaburary
vocab_dict_nirvana = dict()

lyrics_nirvana = lyrics_metallica_nirvana[lyrics_metallica_nirvana["artist"]=="nirvana"]

for lyrics in lyrics_nirvana['lyrics']:
    for k,v in Counter(lyrics).items():
        if k in vocab_dict_nirvana.keys():
            vocab_dict_nirvana[k] += v
        else:
            vocab_dict_nirvana.update({k:v})

# Metallica Vocaburary
vocab_dict_metallica = dict()

lyrics_metallica = lyrics_metallica_nirvana[lyrics_metallica_nirvana["artist"]=="metallica"]

for lyrics in lyrics_metallica['lyrics']:
    for k,v in Counter(lyrics).items():
        if k in vocab_dict_metallica.keys():
            vocab_dict_metallica[k] += v
        else:
            vocab_dict_metallica.update({k:v})

lyrics_metallica_nirvana["lyrics"] = lyrics_metallica_nirvana['lyrics'].apply(remove_low_freq_words)

# Create doc term matrix
v = DictVectorizer()
X_docterm = v.fit_transform(Counter(X) for X in lyrics_metallica_nirvana['lyrics'])
# v.vocabulary_

X_train, X_test, y_train, y_test = model_selection.train_test_split(X_docterm, lyrics_metallica_nirvana['artist'], test_size=0.33, stratify=lyrics_metallica_nirvana['artist'])

# Encode class labels
Encoder = LabelEncoder()
y_train = Encoder.fit_transform(y_train)
y_test = Encoder.fit_transform(y_test)

# # Tf Idf Approach
# tfidf_vect = TfidfVectorizer(lowercase=False)
# #Tfidf_vect.fit(corpus)
# #tfidf_vect.fit(lyrics_data['lyrics'])
# tfidf_vect.fit(X_train)
# X_train_tfidf = tfidf_vect.transform(X_train)
# X_test_tfidf = tfidf_vect.transform(X_test)

# Naive Bayes
from sklearn.naive_bayes import MultinomialNB
clf = MultinomialNB().fit(X_train, y_train)
# clf = MultinomialNB().fit(X_train_tfidf, y_train) # tf-idf

# Predict
predicted = clf.predict(X_test)
# predicted = clf.predict(X_test_tfidf) # tf-idf

# Performance
print("accuracy_score = ", accuracy_score(predicted, y_test))
print("recall_score = ", recall_score(predicted, y_test))
print("precision_score = ", precision_score(predicted, y_test))
print("f1_score = ", f1_score(predicted, y_test))
print("confusion_matrix = \n", confusion_matrix(predicted, y_test))