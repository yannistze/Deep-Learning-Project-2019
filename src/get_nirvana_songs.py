import pandas as pd
import numpy as np

from bs4 import BeautifulSoup
import requests
import json
import string
import ast

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

lyrics_data_nirvana = list()

artist ="nirvana"
album_dict = ast.literal_eval(songs(artist))['albums']

for album, song_list in album_dict.items():
    # print(album, song_list)
    for song in song_list:
        try:
            lyrics_data_nirvana.append([artist, song, lyrics(artist, song)[0]])
        except:
            print(song)

nirvana_df = pd.DataFrame(lyrics_data_nirvana, columns=['artist', 'song', 'lyrics'])
nirvana_df['lyrics'] = nirvana_df.lyrics.str.replace('\t', ' ', regex=False)
nirvana_df.to_csv('nirvana_df.tsv', sep='\t')

# Below used for generating hedwig format data
#
# # Find bracketed song structure words; such words as "[Chorus:]", "[Verse 1:]", etc...
# # Find all the patterns "[*****]"
# def bracketed_song_structure_word_search(sentence):
#     import re
#     match = re.findall(r'\[(.+)\]', sentence)
#     if match:
#         return match
#     else:
#         return None
#
# # Find non-bracketed song structure words; such words as "Chorus:", "Verse 1:", etc...
# # Find all the patterns "******:"
# def song_structure_word_search(sentence):
#     import re
#     match = re.findall(r'(.+):', sentence)
#     if match:
#         return match
#     else:
#         return None
#
# # Find bracketed chorus words
# # Find all the patterns "(*****)"
# def bracketed_chorus_word_search(sentence):
#     import re
#     match = re.findall(r'\((.+)\)', sentence)
#     if match:
#         return match
#     else:
#         return None
#
# # Process lyrics
# def lyrics_processor(sentence):
#     if str(sentence) == 'nan':
#         return None
#     else:
#         # Remove tab if exists
#         sentence = sentence.replace('\t', ' ')
#         # Trim sentence
#         sentence = sentence.strip()
#
#         # Remove bracketed song structure words; such words as "[Chorus:]", "[Verse 1:]", etc...
#         if bracketed_song_structure_word_search(sentence):
#             for s in bracketed_song_structure_word_search(sentence):
#                 s = '[' + s + ']'
#                 sentence = sentence.replace(s, '').strip()
#
#         # Remove non-bracketed song structure words; such words as "Chorus:", "Verse 1:", etc...
#         if song_structure_word_search(sentence):
#             for s in song_structure_word_search(sentence):
#                 s = s + ':'
#                 sentence = sentence.replace(s, '').strip()
#
#         # Remove bracketed chorus words
#         if bracketed_chorus_word_search(sentence):
#             for s in bracketed_chorus_word_search(sentence):
#                 s = '(' + s + ')'
#                 sentence = sentence.replace(s, '').strip()
#
#         # remove "\n\n"
#         sentence = sentence.replace('\n\n', '.')
#         # remove "\n"
#         sentence = sentence.replace('\n', ' ')
#
#         return sentence
#
# # Function to create dictionary of unique genres and corresponding one hot vectors from dataset
# def create_genre_dict(df):
#     genres = df['genre'].unique()
#     genre_dict = dict()
#     for i in range(len(genres)):
#         genre = genres[i]
#         one_hot_vector = np.zeros(len(genres), dtype=int)
#         one_hot_vector[i] = 1
#         one_hot_vector = list(map(lambda x: str(x), one_hot_vector))
#         one_hot_vector = ('').join(one_hot_vector)
#         class_label = i
#         genre_dict.update({str(genre): [one_hot_vector, class_label]})
#     return genre_dict
#
#
# # Function to obtain one hot vector for genre
# def get_one_hot_vector_genre(genre):
#     return genre_dict[genre][0]
#
#
# # Function to obtain integer class label for genre
# def get_class_label(genre):
#     return genre_dict[genre][1]
#
#
# # Load preprocessed lyrics data
# lyrics_data = pd.read_csv('./data/genre_dataframe.csv', usecols=['song', 'artist', 'lyrics'])
# # lyrics_data = lyrics_data[lyrics_data["genre"].notnull()]
# # lyrics_data = lyrics_data[lyrics_data["genre"]!='Other']
#
# # Process lyrics string
# lyrics_data["lyrics"] = lyrics_data["lyrics"].apply(lyrics_processor)
# # lyrics_data = lyrics_data[lyrics_data["lyrics"].str.len() >= 50]
#
# # Create dictionary of unique genres and corresponding one hot vectors
# genre_dict = create_genre_dict(lyrics_data)
#
# # Apply one hot encoding for genre column
# lyrics_data["genre"] = lyrics_data["genre"].apply(str)
# genre = lyrics_data["genre"].apply(get_one_hot_vector_genre)
#
# # Process lyrics string
# lyrics = lyrics_data["lyrics"].apply(lyrics_processor)
#
# # Create new dataframe so that it fit into hedwig format
# df = pd.DataFrame({'genre': genre, 'lyrics': lyrics})
#
# # Save as tsv files
# df.to_csv('./hedwig-data/datasets/Lyrics/df.tsv', sep='\t', index=False, header=False)
