# -*- coding: utf-8 -*-
"""
Spyder Editor
"""

import pandas as pd
import numpy as np

import re
import string
import ast

from tqdm import tqdm

from langdetect import detect_langs

import lyricsgenius



def language_check(dataframe=None):
    """
    Fucntion responsible to check whether a song lyric is in English Language or not.
    """
    
    index_to_remove = []
    
    progress_bar = tqdm(dataframe[~dataframe['lyrics'].isnull()].index.to_list())

    for index in progress_bar:
        
        if isinstance(index, tuple):
            progress_bar.set_description("Processing %s" % index[0] + ' , ' + index[1])
        else:
            progress_bar.set_description("Processing %s" % index)
            
        
        try:
            if isinstance(index, tuple):
                if 'en' not in [item.lang for item in detect_langs(dataframe['lyrics'].loc[index[0]].loc[index[1]])]:
                    index_to_remove.append(index)
            else:
                if 'en' not in [item.lang for item in detect_langs(dataframe['lyrics'].loc[index])]:
                    index_to_remove.append(index)
                
        except:
            index_to_remove.append(index)
            
    
    return index_to_remove
    
    

def create_dataset():
    """
    Dataframe basic creation fuction.
    
    Steps:
        
        - Load the two available datasets
        - Keep only the lyrics written in english language
        - Concatenate the datasets and remove duplicates of the the lyrics
    """
    
    # Load the first dataset and use `song` name and `artist` as a MultiIndex key
    df_1 = pd.read_csv('lyrics.csv', index_col=['song', 'artist'], usecols=['song', 'artist', 'genre', 'lyrics'])
    
    df_1[df_1['genre'] == 'Not Available'] = np.nan
    df_1[df_1['lyrics'] == 'nan'] = np.nan
    
    df_1 = df_1.loc[~df_1.index.duplicated(keep='first')]
    
    index_to_remove = language_check(df_1)
    df_1.drop(index=index_to_remove, inplace=True)
    
    
    # Load the second dataset and use `song` name and `artist`, again, as a MultiIndex key
    df_2 = pd.read_csv('songdata.csv', index_col=['song', 'artist'], usecols=['song', 'artist', 'text'])
    df_2.rename(columns={'text':'lyrics'}, inplace=True)
    # Use the lower case representation of the `song` name and the `artist` as a key to correspond dataset 1 
    df_2.index = df_2.index.map(lambda x: tuple([item.lower() for item in x]))
    
    # Create a `genre` column populated by the np.nan to indicate missing genre for 
    # the second dataset based on the first dataset's guidlines
    df_2[df_2['lyrics'] == 'nan'] = np.nan
    df_2['genre'] = np.nan
    
    df_2 = df_2[['genre', 'lyrics']]
    
    df_2 = df_2.loc[~df_2.index.duplicated(keep='first')]
    
    index_to_remove = language_check(df_2)                        
    df_2.drop(index=index_to_remove, inplace=True)
    
    
    # Concatenate the two datasets in the final dataset
    df = pd.concat([df_1, df_2], axis=0, sort=False)
    # Drop duplicate indices and keep the data from the first occurence of the index
    df = df.loc[~df.index.duplicated(keep='first')] 
    
    df.reset_index(level=[0,1], inplace=True)

    return df_1, df_2, df


def update_dataset(dataframe=None):
    """
    Function responsible to update the dataset's np.nan (empty) lyrics with the ones found on Genius.com.
    """
    
    dataset = dataframe.copy(deep=True)
    
    if type(dataset) != None:
        
        genius = lyricsgenius.Genius(<genius-api-key>, timeout=500*60)    
        genius.remove_section_headers = True
        genius.skip_non_songs = False
        genius.excluded_terms = ["(Remix)", "(Live)"]    
        
        progress_bar = tqdm(dataset[dataset['lyrics'].isnull()].index.to_list())
        
        for index in progress_bar:
        
            progress_bar.set_description("Processing %s" % str(index))
            
            temp = genius.search_song(dataset['song'].loc[index].replace('-', ' '), dataset['artist'].loc[index].replace('-', ' '))
            
            if (temp != None): 
                if '(Script)' not in temp.title:
                    dataset['lyrics'].loc[index] = temp.lyrics.lstrip() 
    
    else:
        raise ValueError('Dataset cannot be None...')
        
    return dataset



def clean_dataset(dataframe=None):
    """
    Function responsible for basic cleaning of the dataframe downloaded from Genius.com.
    """
    
    dataset = dataframe.copy(deep=True)
    
    index_to_remove = language_check(dataset)
    dataset.drop(index=index_to_remove, inplace=True)
    
    pattern = re.compile('[%s]' % re.escape(string.punctuation.replace('[', '').replace(']', '')))
    dataset['lyrics'] = dataset['lyrics'].str.replace(pattern, '')   
   
    dataset = dataset[dataset['lyrics'].str.len() <= 20000]

    return dataset

def final_dataset(dataframe=None):
    """
    Function responsible for creating the final dataset to be used.
    """  
    
    dataset = dataframe.copy(deep=True)
    
    index_to_remove = language_check(dataset)
    dataset.drop(index=index_to_remove, inplace=True)
    
    dataset = dataset[~dataset['lyrics'].isnull()]
    dataset = dataset[~dataset['artist'].isnull()]
    dataset = dataset[~dataset['song'].isnull()]
    
    lyrics_df = dataset[~dataset['lyrics'].isnull()].copy(deep=True)
    genre_df = dataset[~dataset['genre'].isnull()].copy(deep=True)
    
    lyrics_df.drop(columns='genre', inplace=True)
        
    return lyrics_df, genre_df


def statistics(dataframe=None):
    """
    Function to output basic statistics for the dataset.
    """
    
    dataframe.drop(columns=['song', 'lyrics'], inplace=True)
    dataframe['cnt'] = 1
    
    dataframe.set_index(['artist', 'genre'], inplace=True)
    
    artists = dataframe.copy(deep=True)
    artists = artists.groupby(level=0).sum()
    
    genre = dataframe.copy(deep=True)
    genre = genre.groupby(level=1).sum()
    
    return artists, genre
    
    
        
tmp_1, tmp_2, tmp = create_dataset()

tmp.to_csv('dataframe.csv')
tmp_1.to_csv('dataframe_1.csv')
tmp_2.to_csv('dataframe_2.csv')

tmp = update_dataset(tmp)
tmp = clean_dataset(tmp)
    
lyrics, genre = final_dataset(tmp)
genre.to_csv('genre_final_df.csv')
lyrics.to_csv('lyrics_final_df.csv')

#%%testing
#temp_df.to_csv('temp_batch_1.csv', index=True, index_label='index')
#test_tmp = tmp[tmp['lyrics'].isnull()]
#tmp = pd.read_csv('dataframe.csv', index_col=0)
#index_list = tmp.index.to_list()
#temporary_df = tmp[tmp.index.isin(index_list[238341:317788])]
#temporary_df = temporary_df[~temporary_df['song'].isnull()]