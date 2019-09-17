# Deep-Learning-Project
Attribute a song to an author or a genre

## Setup

Our project depends on Hedwig, which is designed for Python 3.6 and [PyTorch](https://pytorch.org/) 0.4. PyTorch recommends [Anaconda](https://www.anaconda.com/distribution/) for managing your environment. We'd recommend creating a custom environment as follows:

```
$ conda create --f environment.yml
```

Code depends on data from NLTK (e.g., stopwords) so you'll have to download them. Run:

```
$ python src/nltk_download.py
```



## Datasets

Download the Reuters, word2vec embeddings from [`hedwig-data`](https://git.uwaterloo.ca/jimmylin/hedwig-data).

```
$ git clone https://github.com/j-cahill/hedwig.git
$ git clone https://git.uwaterloo.ca/jimmylin/hedwig-data.git
```

After cloning the hedwig-data repo, you need to unzip the embeddings and run the preprocessing script:

```
cd hedwig-data/embeddings/word2vec 
gzip -d GoogleNews-vectors-negative300.bin.gz 
python bin2txt.py GoogleNews-vectors-negative300.bin GoogleNews-vectors-negative300.txt 
```

## Create Lyrics Dataset

Download [380,000 lyrics from metrolyrics](https://kaggle.com/gyani95/380000-lyrics-from-metrolyrics) and [55,000+ song lyrics](https://www.kaggle.com/mousehead/songlyrics) from Kaggle. Run

```
$ python src/data_preprocessing.py 
```

To impute the missing genre values using the Genius API. Then run

```
$ python src/lyrics_preprocessor.py
$ python src/lyrics_preprocessor_artists.py
```

to perform basic data cleanup and create train/test/dev splits for both genre and artist.

## Folder Structure

It is imperative for the code to run that the repo be structured in the following way

```
Deep-Learning-Project   
│
|---hedwig
│   
└---hedwig-data
    │----LyricsGenre
    |	train.tsv
    |	test.tsv
    |	dev.tsv
    │----LyricsArtist
    |	train.tsv
    |	test.tsv
	|	dev.tsv
```

Other files and folder may be present as well but this structure must be observed.

# Model Training

## Genre

Train a BERT model with the following command, testing will immediately follow

```
$ cd hedwig/
$ python -m models.bert --dataset LyricsGenre --model bert-base-uncased --max-seq-length 256 --batch-size 16 --lr 2e-5 --epochs 2
```

## Artist 

```
$ cd hedwig/
$ python -m models.bert --dataset LyricsArtist --model bert-base-uncased --max-seq-length 256 --batch-size 16 --lr 2e-5 --epochs 20
```

Models are saved to `hedwig/model_checkpoints`
