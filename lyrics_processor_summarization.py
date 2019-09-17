import pandas as pd
import numpy as np


# Function to create dictionary of unique genres and corresponding one hot vectors from dataset
def create_genre_dict(df):
    genres = df['genre'].unique()
    genre_dict = dict()
    for i in range(len(genres)):
        genre = genres[i]
        one_hot_vector = np.zeros(len(genres), dtype=int)
        one_hot_vector[i] = 1
        one_hot_vector = list(map(lambda x: str(x), one_hot_vector))
        one_hot_vector = ('').join(one_hot_vector)
        class_label = i
        genre_dict.update({str(genre): [one_hot_vector, class_label]})
    return genre_dict


# Function to obtain one hot vector for genre
def get_one_hot_vector_genre(genre):
    return genre_dict[genre][0]


# Function to obtain integer class label for genre
def get_class_label(genre):
    return genre_dict[genre][1]


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



# Load preprocessed lyrics data

# For genre
lyrics_data = pd.read_csv('./data/genre_final.csv', usecols=['song', 'artist', 'genre', 'lyrics'])
lyrics_data = lyrics_data[lyrics_data["genre"].notnull()]
lyrics_data = lyrics_data[lyrics_data["genre"]!='Other']

# For artist
# lyrics_data = pd.read_csv('./data/lyrics_final.csv', usecols=['song', 'artist', 'lyrics'])


# Common procedure for genre/artist
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'\[(.+)\]', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'(.+):', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'\((.+)\)', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\t', ' ')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\n\n', '.')

lyrics_data['lyrics_nchar'] = lyrics_data['lyrics']

lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\n', ' ')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\r', ' ')

# For genre

# Undersampling of lyrics data; each class to have 500
undersample_size = min(lyrics_data.groupby(['genre']).count()['song'])
# undersample_size = 500
replace = True
fn = lambda obj: obj.loc[np.random.choice(obj.index, undersample_size, replace),:]
lyrics_data = lyrics_data.groupby('genre', as_index=False).apply(fn)

# Create dictionary of unique genres and corresponding one hot vectors
genre_dict = create_genre_dict(lyrics_data)

# Apply one hot encoding for genre column
lyrics_data["genre_encoded"] = lyrics_data["genre"].apply(str)
lyrics_data["genre_encoded"] = lyrics_data["genre_encoded"].apply(get_one_hot_vector_genre)

####################################################################################################################################
# For Artist
####################################################################################################################################

# num_of_artists = 100
# top_artists = list(lyrics_data.groupby(['artist']).count().sort_values(['song'], ascending=False).head(num_of_artists).index)
# lyrics_data = lyrics_data[lyrics_data["artist"].isin(top_artists)]
#
# # Create dictionary of unique genres and corresponding one hot vectors
# artist_dict = create_artist_dict(lyrics_data)
#
# # Apply one hot encoding for genre column
# lyrics_data["artist_encoded"] = lyrics_data["artist"].apply(str)
# lyrics_data["artist_encoded"] = lyrics_data["artist_encoded"].apply(get_one_hot_vector_artist)


lyrics_data = lyrics_data[lyrics_data['lyrics'].str.strip() != ""]
lyrics_data = lyrics_data[lyrics_data['lyrics'].str.len() >= 200]

# lyrics_data.to_csv('./data/lyrics_final_for_summarization.tsv', sep='\t', index=False, header=True)
lyrics_data.to_csv('./data/genre_final_for_summarization.tsv', sep='\t', index=False, header=True)



# # For test
#
# df = pd.read_csv('./summarized_data/genre.tsv', sep='\t', names=["genre","lyrics"], dtype=str)
# df["genre"] = df["genre"].str.zfill(10)
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.2, stratify=df["genre"])
# dev, test = train_test_split(test, test_size=0.5, stratify=test["genre"])
#
# train.to_csv('./summarized_data/text_rank/train.tsv', sep='\t', index=False, header=False)
# test.to_csv('./summarized_data/text_rank/test.tsv', sep='\t', index=False, header=False)
# dev.to_csv('./summarized_data/text_rank/dev.tsv', sep='\t', index=False, header=False)
#
#
# df = pd.read_csv('./summarized_data/genre_2.tsv', sep='\t', names=["genre","lyrics"], dtype=str)
# df["genre"] = df["genre"].str.zfill(10)
# df = df[df["genre"].str.len() == 10]
# print(df.groupby(["genre"]).count())
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.2, stratify=df["genre"])
# dev, test = train_test_split(test, test_size=0.5, stratify=test["genre"])
#
# train.to_csv('./summarized_data/lex_rank/genre/train.tsv', sep='\t', index=False, header=False)
# test.to_csv('./summarized_data/lex_rank/genre/test.tsv', sep='\t', index=False, header=False)
# dev.to_csv('./summarized_data/lex_rank/genre/dev.tsv', sep='\t', index=False, header=False)
#
#
#
# df = pd.read_csv('./summarized_data/lyrics_2.tsv', sep='\t', names=["artist","lyrics"], dtype=str)
# df["artist"] = df["artist"].str.zfill(10)
# df = df[df["artist"].str.len() == 100]
# print(df.groupby(["artist"]).count())
# from sklearn.model_selection import train_test_split
# train, test = train_test_split(df, test_size=0.2, stratify=df["artist"])
# dev, test = train_test_split(test, test_size=0.5, stratify=test["artist"])
#
# train.to_csv('./summarized_data/lex_rank/artist/train.tsv', sep='\t', index=False, header=False)
# test.to_csv('./summarized_data/lex_rank/artist/test.tsv', sep='\t', index=False, header=False)
# dev.to_csv('./summarized_data/lex_rank/artist/dev.tsv', sep='\t', index=False, header=False)
#
# train, test = train_test_split(df, test_size=0.2, stratify=df["artist"])
# dev, test = train_test_split(test, test_size=0.5, stratify=test["artist"])
