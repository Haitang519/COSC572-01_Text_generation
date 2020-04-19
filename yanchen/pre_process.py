import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import re
import numpy as np
from constractions import contractions

VALIDATION_SPLIT = 0.1

def clean_plot(plot, remove_stopwords, lemma):
    # lowercase
    plot = plot.lower()
    # remove cite notation eg.[1]
    plot = re.sub('\[\d*\]', '', plot)
    plot = re.sub('\ \ *', ' ', plot)
    # remove non-letter characters
    plot = re.sub(r"[^a-zA-Z]+", r" ", plot)

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


def get_data(plots, remove_stopwords=True, lemma=True):
    data = []
    word_index = {}
    index = 0
    lens = []
    count = 0
    word_count ={}
    for plot in plots['Plot']:
        count +=1
        length = 0
        plot = clean_plot(plot, remove_stopwords=False, lemma=True)
        data.append(plot)
        plot_words = plot.strip().split(' ')
        
        for word in plot_words:
            length += 1
            if word not in word_index:
                word_count[word]=1
            else:
                word_count[word]+=1
            if word not in word_index:
                word_index[word] = index
                index += 1
            
        lens.append(length)
        if count%1000==0:
            print(count)

    # calculate the upper bound of length of sentence
    lens.sort()
    n = len(lens)
    Q1 = lens[int((n+1)/4)]
    Q3 = lens[int(3*(n + 1)/4)]
    MAX_SEQUENCE_LENGTH = int(Q3 + 1.5 * (Q3 - Q1))
    # MAX_SEQUENCE_LENGTH = int(Q3)

    return data, word_index,word_count, MAX_SEQUENCE_LENGTH







plots = pd.read_csv("wiki_movie_plots_deduped.csv", sep=',', encoding='latin1')

data, word_index,word_count, MAX_SEQUENCE_LENGTH = get_data(plots)

vocab = {k: v for k, v in word_count.items() if v>20}
vocab_index = {k:i for (k,i) in zip(vocab.keys(),range(len(vocab)))}
# vocab_index['UNK'] = len(vocab_index)

data1 = []
count = 0
for sent in data:
    count +=1
    plot_words = sent.strip().split(' ')
    data1.append(" ".join([w if w in vocab_index else 'UNK' for w in plot_words]))
    if count%1000==0:
        print(count)
        
tokenizer = Tokenizer()
tokenizer.fit_on_texts(data1)

sequences = tokenizer.texts_to_sequences(data1)
word_index = tokenizer.word_index
word_index['PAD'] = 0
print('Found %s unique tokens.' % len(word_index))

# word_index = tokenizer.word_index
encode_data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
# for i in range(len(sequences)):
#     if max(sequences[i])>=23415:
#         print(max(encode_data[i]))
#         print(i)

arr = np.array(encode_data)

# split the data into a training set and a validation set
indices = np.arange(arr.shape[0])
np.random.shuffle(indices)
arr = arr[indices]
num_validation_samples = int(VALIDATION_SPLIT * arr.shape[0])

train_data = arr[:-num_validation_samples]
val_data = arr[-num_validation_samples:]


np.savez_compressed('save/data1.npz', train=train_data, val=val_data, length=MAX_SEQUENCE_LENGTH)
f = open('save/word_index1.txt', 'w')
f.write(str(word_index))
f.close()


data = np.load('save/data1.npz')

f = open('save/word_index1.txt', 'r')
word_dict = eval(f.read())
f.close()
