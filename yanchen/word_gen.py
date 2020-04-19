import argparse
import os, random
from collections import Counter

from keras.models import Sequential,load_model
from keras.layers import Embedding, Bidirectional, LSTM, Dense, TimeDistributed, Dropout
from keras.callbacks import LambdaCallback
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
import numpy as np
from keras import backend as K
from keras.initializers import Constant
import pandas as pd 
import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from keras.callbacks import ModelCheckpoint, Callback
# from google.colab import drive
import nltk
nltk.download('wordnet')
  

UNK = '[UNK]'
PAD = '[PAD]'
START = '<s>'
END = '</s>'
vocab = Counter()
data = []

contractions = {
    "ain't": "am not",
    "aren't": "are not",
    "can't": "cannot",
    "can't've": "cannot have",
    "'cause": "because",
    "could've": "could have",
    "couldn't": "could not",
    "couldn't've": "could not have",
    "didn't": "did not",
    "doesn't": "does not",
    "don't": "do not",
    "hadn't": "had not",
    "hadn't've": "had not have",
    "hasn't": "has not",
    "haven't": "have not",
    "he'd": "he would",
    "he'd've": "he would have",
    "he'll": "he will",
    "he's": "he is",
    "how'd": "how did",
    "how'll": "how will",
    "how's": "how is",
    "i'd": "i would",
    "i'll": "i will",
    "i'm": "i am",
    "i've": "i have",
    "isn't": "is not",
    "it'd": "it would",
    "it'll": "it will",
    "it's": "it is",
    "let's": "let us",
    "ma'am": "madam",
    "mayn't": "may not",
    "might've": "might have",
    "mightn't": "might not",
    "must've": "must have",
    "mustn't": "must not",
    "needn't": "need not",
    "oughtn't": "ought not",
    "shan't": "shall not",
    "sha'n't": "shall not",
    "she'd": "she would",
    "she'll": "she will",
    "she's": "she is",
    "should've": "should have",
    "shouldn't": "should not",
    "that'd": "that would",
    "that's": "that is",
    "there'd": "there had",
    "there's": "there is",
    "they'd": "they would",
    "they'll": "they will",
    "they're": "they are",
    "they've": "they have",
    "wasn't": "was not",
    "we'd": "we would",
    "we'll": "we will",
    "we're": "we are",
    "we've": "we have",
    "weren't": "were not",
    "what'll": "what will",
    "what're": "what are",
    "what's": "what is",
    "what've": "what have",
    "where'd": "where did",
    "where's": "where is",
    "who'll": "who will",
    "who's": "who is",
    "won't": "will not",
    "wouldn't": "would not",
    "you'd": "you would",
    "you'll": "you will",
    "you're": "you are"
}

def clean_plot(plot, remove_stopwords=False, lemma=True):
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

def get_vocabulary_and_data(plots, max_vocab_size=20000):
    vocab = Counter()
    data = []
    lens = []
    count = 0
    for line in plots.Plot:
        if count%1000==0:
            print(count)
        count+=1
        line = clean_plot(line)
        sent = [START]
        vocab[START]+=1
        vocab[END]+=1
        lens.append(len(line.strip().split(' ')))
        for tok in line.strip().split(' '):
            sent.append(tok)
            vocab[tok]+=1
        sent.append(END)
        data.append(sent)
    # if max_vocab_size:
    vocab = {k: v for k, v in sorted(vocab.items(), key=lambda item: item[1],reverse=True)}
    vocab = list(vocab)
    vocab = vocab[:max_vocab_size]
    vocab = [UNK, PAD] + vocab
    n = len(lens)
    Q1 = lens[int((n+1)/4)]
    Q3 = lens[int(3*(n + 1)/4)]
    MAX_SEQUENCE_LENGTH = int(Q3 + 0.5 * (Q3 - Q1))

    return {k:v for v,k in enumerate(vocab)}, data,MAX_SEQUENCE_LENGTH



def vectorize_sequence(seq, vocab):
    seq = [tok if tok in vocab else UNK for tok in seq]
    return [vocab[tok] for tok in seq]


def unvectorize_sequence(seq, vocab):
    translate = sorted(vocab.keys(),key=lambda k:vocab[k])
    return [translate[i] for i in seq]


def one_hot_encode_label(label, vocab):
    vec = [1.0 if l==label else 0.0 for l in vocab]
    return vec


def batch_generator_lm(data, vocab, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for sent in data:
            batch_x.append(vectorize_sequence(sent, vocab))
            batch_y.append([one_hot_encode_label(token, vocab) for token in shift_by_one(sent)])
            if len(batch_x) >= batch_size:
                # Pad Sequences in batch to same length
                batch_x = pad_sequences(batch_x, vocab[PAD])
                batch_y = pad_sequences(batch_y, one_hot_encode_label(PAD, vocab))
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []


def describe_data(data, generator):
    batch_x, batch_y = [], []
    for bx, by in generator:
        batch_x = bx
        batch_y = by
        break
    print('Data example:',data[0])
    print('Batch input shape:', batch_x.shape)
    print('Batch output shape:', batch_y.shape)


def pad_sequences(batch_x, pad_value):
    ''' This function should take a batch of sequences of different lengths
        and pad them with the pad_value token so that they are all the same length.

        Assume that batch_x is a list of lists.
    '''
    pad_length = len(max(batch_x, key=lambda x: len(x)))
    for i, x in enumerate(batch_x):
        if len(x) < pad_length:
            batch_x[i] = x + ([pad_value] * (pad_length - len(x)))

    return batch_x


def generate_text(language_model, vocab):
    prediction = [START]
    while not (prediction[-1] == END or len(prediction)>=100):
        next_token_one_hot = language_model.predict(np.array([[vocab[p] for p in prediction]]), batch_size=1)[0][-1]
        threshold = random.random()
        sum = 0
        next_token = 0
        for i,p in enumerate(next_token_one_hot):
            sum += p
            if sum>threshold:
                next_token = i
                break
        for w, i in vocab.items():
            if i==next_token:
                prediction.append(w)
                break
    return prediction

# TODO
def load_pretrained_embeddings(glove_file, vocab):
    embedding_matrix = np.zeros((len(vocab), 100))
    with open(glove_file, encoding='utf8') as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, 'f', sep=' ')
            # Each line will be a word and a list of floats, separated by spaces.
            # If the word is in your vocabulary, create a numpy array from the list of floats.
            # Assign the array to the correct row of embedding_matrix.
            if word in vocab:
                embedding_matrix[vocab[word]] = coefs
    embedding_matrix[vocab[UNK]] = np.random.randn(100)
    return embedding_matrix


def shift_by_one(seq):
    '''
    input: ['<s>', 'The', 'dog', 'chased', 'the', 'cat', 'around', 'the', 'house', '</s>']
    output: ['The', 'dog', 'chased', 'the', 'cat', 'around', 'the', 'house', '</s>', '[PAD]']
    '''
    result = seq[1:]
    result.append('[PAD]')
    return result
    

def clean_data(data,vocab,MAX_SEQUENCE_LENGTH):
    data1 = []
    count = 0
    for line in data:
        if count%1000==0:
            print(count)
        count+=1
        if len(line)>MAX_SEQUENCE_LENGTH:
            line1 = line[:MAX_SEQUENCE_LENGTH]
            line1.append(END)
            data1.append(line1)
        else: 
            data1.append(line)
    
    train_data = []
    count = 0
    for sent in data1:
        count +=1
        [w if w in vocab else 'UNK' for w in sent]
        train_data.append([w if w in vocab else 'UNK' for w in sent])
        if count%1000==0:
            print(count)
    return train_data

def on_epoch_end(epoch, _):
    print()
    print('----- Generating text after Epoch: %d' % epoch)
    for i in range(10):
        prediction = [START]
        while not (prediction[-1] == END or len(prediction)>=100):
            next_token_one_hot = language_model.predict(np.array([[vocab[p] for p in prediction]]), batch_size=1)[0][-1]
            threshold = random.random()
            sum = 0
            next_token = 0
            for i,p in enumerate(next_token_one_hot):
                sum += p
                if sum>threshold:
                    next_token = i
                    break
            for w, i in vocab.items():
                if i==next_token:
                    prediction.append(w)
                    break
        sys.stdout.write(prediction)
        sys.stdout.flush()
# drive.mount('/content/gdrive/')
# glove_file = '/content/gdrive/My Drive/Colab Notebooks/glove.6B.100d.txt'
# data_path = '/content/gdrive/My Drive/Colab Notebooks/wiki_movie_plots_deduped.csv'
glove_file = 'glove.6B.100d.txt'
data_path = 'wiki_movie_plots_deduped.csv'
plots = pd.read_csv(data_path, sep=',', encoding='latin1')
train_plots, test_plots = train_test_split(plots,test_size = 0.15)
dev_plots, test_plots = train_test_split(test_plots,test_size = 0.5)

vocab, train_data, MAX_SEQUENCE_LENGTH = get_vocabulary_and_data(train_plots)
_, dev_data,_ = get_vocabulary_and_data(dev_plots)
_, test_data,_ = get_vocabulary_and_data(test_plots)

train_data = clean_data(train_data,vocab,MAX_SEQUENCE_LENGTH)
dev_data = clean_data(dev_data,vocab,MAX_SEQUENCE_LENGTH)
test_data = clean_data(test_data,vocab,MAX_SEQUENCE_LENGTH)

embedding_matrix = load_pretrained_embeddings(glove_file, vocab)
describe_data(train_data, batch_generator_lm(train_data, vocab, 10))
weights = ModelCheckpoint(filepath = 'model10.h5')
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)


language_model = Sequential()
language_model.add(Embedding(len(vocab), 100,embeddings_initializer=Constant(embedding_matrix),trainable=False))
language_model.add(Dropout(0.2))
language_model.add(LSTM(100, return_sequences=True))
language_model.add(Dropout(0.2))
language_model.add(TimeDistributed(Dense(len(vocab), activation='softmax')))

language_model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])



if not os.path.exists('model10.h5'):
    language_model.fit_generator(batch_generator_lm(train_data, vocab,10),
                             epochs=60, steps_per_epoch=len(train_data)/10,callbacks=[print_callback,weights])
else:
    language_model = load_model('model10.h5')
    print(language_model.summary())
    language_model.fit_generator(batch_generator_lm(train_data, vocab,10),
                             epochs=60, steps_per_epoch=len(train_data)/10,callbacks=[print_callback,weights])

# Evaluation
loss, acc = language_model.evaluate_generator(batch_generator_lm(dev_data, vocab),
                                              steps=len(dev_data))
print('Dev Loss:', loss, 'Dev Acc:', acc)
loss, acc = language_model.evaluate_generator(batch_generator_lm(test_data, vocab),
                                              steps=len(test_data))

    

