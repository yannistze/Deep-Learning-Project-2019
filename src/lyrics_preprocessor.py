import pandas as pd
import numpy as np


# Function to create dictionary of unique genres and corresponding one hot vectors from dataset (for Hedwig to consume)
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


# Load preprocessed lyrics data
lyrics_data = pd.read_csv('./data/genre_final.csv', usecols=['song', 'artist', 'genre', 'lyrics'])
lyrics_data = lyrics_data[lyrics_data["genre"].notnull()]
lyrics_data = lyrics_data[lyrics_data["genre"]!='Other']

# Remove non-lyrics words and characters
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'\[(.+)\]', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'(.+):', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace(r'\((.+)\)', '')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\t', ' ')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\n\n', '.')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\n', ' ')
lyrics_data['lyrics'] = lyrics_data['lyrics'].str.replace('\r', ' ')

# Create dictionary of unique genres and corresponding one hot vectors
genre_dict = create_genre_dict(lyrics_data)

# Save genre / one-hot vector dictionary
genre_dict_df = pd.DataFrame.from_dict(genre_dict, orient='index')
genre_dict_df.to_csv('./data/genre_dict_df.tsv', sep='\t', index=True, header=False)

# Apply one hot encoding for genre column
lyrics_data["genre"] = lyrics_data["genre"].apply(str)
genre = lyrics_data["genre"].apply(get_one_hot_vector_genre)

# Process lyrics string
lyrics = lyrics_data["lyrics"]

# Create new dataframe so that it fit into hedwig format
df = pd.DataFrame({'genre': genre, 'lyrics': lyrics})
df = df[df['lyrics'].str.strip() != ""]
df = df[df['lyrics'].str.len() >= 200]

df.to_csv('./data/genre_final_processed.tsv', sep='\t', index=False, header=False)

# Split into train, test, validation datasets, stratified by genre column
# Split at [0:80], [80:90], [90:100] respectively
from sklearn.model_selection import train_test_split
train, test = train_test_split(df, test_size=0.2, stratify=df["genre"])
dev, test = train_test_split(test, test_size=0.5, stratify=test["genre"])

# Save as tsv files
train.to_csv('./hedwig-data/datasets/LyricsGenre/train.tsv', sep='\t', index=False, header=False)
test.to_csv('./hedwig-data/datasets/LyricsGenre/test.tsv', sep='\t', index=False, header=False)
dev.to_csv('./hedwig-data/datasets/LyricsGenre/dev.tsv', sep='\t', index=False, header=False)


# # Check 3 files
#
# df = pd.read_csv('./hedwig-data/datasets/LyricsGenre/train.tsv', sep='\t', names=["genre","lyrics"])
# print(df.groupby(['genre']).count())
# print(len(df['genre']))
# print(df['genre'].unique())
# print(len(df['genre'].unique()))
# print(len(df))
#
# df = pd.read_csv('./hedwig-data/datasets/LyricsGenre/test.tsv', sep='\t', names=["genre","lyrics"])
# print(df.groupby(['genre']).count())
# print(len(df['genre']))
# print(df['genre'].unique())
# print(len(df['genre'].unique()))
# print(len(df))
#
# df = pd.read_csv('./hedwig-data/datasets/LyricsGenre/dev.tsv', sep='\t', names=["genre","lyrics"])
# print(df.groupby(['genre']).count())
# print(len(df['genre']))
# print(df['genre'].unique())
# print(len(df['genre'].unique()))
# print(len(df))
