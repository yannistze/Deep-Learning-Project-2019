import pandas as pd
import numpy as np


# Function to create dictionary of unique genres and corresponding one hot vectors from dataset (for Hedwig to consume)
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
lyrics_data = pd.read_csv('./data/lyrics_final.csv', usecols=['song', 'artist', 'lyrics'])

# Remove non-lyrics words and characters
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'\[(.+)\]', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'(.+):', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'\((.+)\)', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\t', ' ')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\n\n', '.')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\n', ' ')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\r', ' ')

# Limit the number of artists to 25, 100 etc.
num_of_artists = 100
top_artists = list(lyrics_data.groupby(['artist']).count().sort_values(['song'], ascending=False).head(num_of_artists).index)
lyrics_data = lyrics_data[lyrics_data["artist"].isin(top_artists)]

# Create dictionary of unique genres and corresponding one hot vectors
artist_dict = create_artist_dict(lyrics_data)

# Save artist / one-hot vector dictionary
artist_dict_df = pd.DataFrame.from_dict(artist_dict, orient='index')
artist_dict_df.to_csv('./data/artist_dict_df.tsv', sep='\t', index=True, header=False)

# Apply one hot encoding for genre column
lyrics_data["artist"] = lyrics_data["artist"].apply(str)
artist = lyrics_data["artist"].apply(get_one_hot_vector_artist)

# Process lyrics string
lyrics = lyrics_data["lyrics"]

# Create new dataframe so that it fit into hedwig format
df = pd.DataFrame({'artist': artist, 'lyrics': lyrics})
df = df[df['lyrics'].str.strip() != ""]
df = df[df['lyrics'].str.len() >= 200]

df.to_csv('./data/lyrics_final_processed.tsv', sep='\t', index=False, header=False)

# Split into train, test, validation datasets, stratified by artist column
# Split at [0:80], [80:90], [90:100] respectively
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, stratify=df["artist"])
dev, test = train_test_split(test, test_size=0.5, stratify=test["artist"])

# Save as tsv files
train.to_csv('./hedwig-data/datasets/LyricsArtist/train.tsv', sep='\t', index=False, header=False)
test.to_csv('./hedwig-data/datasets/LyricsArtist/test.tsv', sep='\t', index=False, header=False)
dev.to_csv('./hedwig-data/datasets/LyricsArtist/dev.tsv', sep='\t', index=False, header=False)


# # Check 3 files
#
# df = pd.read_csv('./hedwig-data/datasets/LyricsArtist/train.tsv', sep='\t', names=["artist","lyrics"])
# print(df.groupby(['artist']).count())
# print(len(df['artist']))
# print(df['artist'].unique())
# print(len(df['artist'].unique()))
# print(len(df))
#
# df = pd.read_csv('./hedwig-data/datasets/LyricsArtist/test.tsv', sep='\t', names=["artist","lyrics"])
# print(df.groupby(['artist']).count())
# print(len(df['artist']))
# print(df['artist'].unique())
# print(len(df['artist'].unique()))
# print(len(df))
#
# df = pd.read_csv('./hedwig-data/datasets/LyricsArtist/dev.tsv', sep='\t', names=["artist","lyrics"])
# print(df.groupby(['artist']).count())
# print(len(df['artist']))
# print(df['artist'].unique())
# print(len(df['artist'].unique()))
# print(len(df))
