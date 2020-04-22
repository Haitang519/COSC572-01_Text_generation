from collections import Counter

import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
from constractions import contractions


VALIDATION_SPLIT = 0.1
PAD = '[PAD]'
END = '[END]'
UNI = '[UNI]'


def clean_plot(plot, remove_stopwords, lemma):
    # lowercase
    plot = plot.lower()
    # remove cite notation eg.[1]
    plot = re.sub('\[\d*\]', '', plot)
    plot = re.sub('\ \ *', ' ', plot)
    # remove non-letter characters
    # plot = re.sub(r"[^a-zA-Z]+", r" ", plot)
    r1 = r'(,|\.|/|;|\'|`|\[|\]|<|>|\?|:|"|\{|\}|\~|!|@|#|\$|%|\^|&|\(|\)|-|=|\_|\+|，|。|、|；|‘|’|【|】|·|！| |…|（|）)'
    plot = re.split(r1, plot)
    new_plot = []
    for token in plot:
        if token != " ":
            new_plot.append(token)
    plot = " ".join(new_plot)

    # expand contraction
    wnl = WordNetLemmatizer()
    plot = plot.strip().split(' ')
    new_plot = []
    for word in plot:
        if word in contractions:
            new_plot.append(contractions[word])
        else:
            if lemma:
                word = wnl.lemmatize(word)
            new_plot.append(word)
    plot = " ".join(new_plot)

    # remove stopwords
    if remove_stopwords:
        plot = plot.split()
        stops = set(stopwords.words("english"))
        plot = [w for w in plot if not w in stops]
        plot = " ".join(plot)
    return plot


def get_data(df_plots, remove_stopwords=False, lemma=True):
    vocabs = Counter()
    plots_list = []
    for plot in df_plots['Plot']:
        plot = clean_plot(plot, remove_stopwords, lemma)
        plot = plot + ' ' + END
        plots_list.append(plot)
        tokens = plot.strip().split(' ')
        for token in tokens:
            vocabs[token] += 1

    data = []
    lens = []
    for plot in plots_list:
        length = 0
        plot_words = plot.strip().split(' ')
        new_plot = []
        for word in plot_words:
            length += 1
            if vocabs[word] > 15:
                new_plot.append(word)
            else:
                new_plot.append(UNI)
        data.append(" ".join(new_plot))
        lens.append(length)


    # calculate the upper bound of length of sentence
    lens.sort()
    n = len(lens)
    Q1 = lens[int((n+1)/4)]
    Q3 = lens[int(3*(n + 1)/4)]
    MAX_SEQUENCE_LENGTH = int(Q3 + 1.5 * (Q3 - Q1))
    # MAX_SEQUENCE_LENGTH = int(Q3)

    return data


if __name__ == "__main__":
    df_plots = pd.read_csv("wiki_movie_plots_deduped.csv", sep=',', encoding='latin1')

    data = get_data(df_plots)
    # print(data)
    MAX_SEQUENCE_LENGTH = 300

    tokenizer = Tokenizer(filters='()\t\n', oov_token='[UNK]')
    tokenizer.fit_on_texts(data)
    sequences = tokenizer.texts_to_sequences(data)
    # print(sequences)

    word_index = tokenizer.word_index
    word_index[PAD] = 0
    # print(word_index)
    print('Found %s unique tokens.' % len(word_index))

    encode_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)

    arr = np.array(encode_data)

    # split the data into a training set and a validation set
    indices = np.arange(arr.shape[0])
    np.random.shuffle(indices)
    arr = arr[indices]
    num_validation_samples = int(VALIDATION_SPLIT * arr.shape[0])

    train_data = arr[:-num_validation_samples]
    val_data = arr[-num_validation_samples:]

    # save data into file
    np.savez_compressed('save/data.npz', train=train_data, val=val_data, length=MAX_SEQUENCE_LENGTH, data=np.array(data))
    f = open('save/word_index.txt', 'w')
    f.write(str(word_index))
    f.close()
