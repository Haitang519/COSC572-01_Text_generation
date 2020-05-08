from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout, TimeDistributed
from keras.layers import LSTM
from keras.optimizers import RMSprop
from keras.utils.data_utils import get_file
from keras.callbacks import ModelCheckpoint, Callback
import numpy as np
import random
import sys
import io
import re
import os
import argparse

import pandas as pd 
from collections import Counter
from sklearn.model_selection import train_test_split


args = argparse.ArgumentParser(description='Program description.')
args.add_argument('-lr','--learning-rate', default=0.01, type=float, help='Learning rate')
args.add_argument('-e','--epochs', default=10, type=int, help='Number of epochs')
args.add_argument('-do','--dropout', default=0.2, type=int, help='Dropout rate')
args.add_argument('-hs','--hidden-size', default=1000, type=int, help='Hidden layer size')
args.add_argument('-b','--batch-size', default=30, type=int, help='Maximum character sequence length')
args.add_argument('-m','--maxlen', default=40, type=int, help='Batch Size')
args = args.parse_args()


data_path = '../data/wiki_movie_plots_deduped.csv'
df = pd.read_csv(data_path)
vocab = Counter()
data = []

for line in df.Plot:
    # remove citation
    line = line.lower()
    line = re.sub('\[\d*\]','',line)
    line = re.sub('\ \ *',' ',line)
    # remove format
    line = re.sub(r"won't", "will not", line)
    line = re.sub(r"can't", "can not", line)
    line = re.sub(r"'re", " are", line)
    line = re.sub(r"'s", " is", line)
    line = re.sub(r"'d", " would", line)
    line = re.sub(r"'ll", " will", line)
    line = re.sub(r"'t", " not", line)
    line = re.sub(r"'ve", " have", line)
    line = re.sub(r"'m", " am", line)
    line = re.sub('\n',' ',line)  
    line = re.sub('\r','',line)  
    sent = ''
    sent = sent+line+'\n'
    data.append(sent)

alphabet = "abcdefghijklmnopqrstuvwxyz0123456789,;.!?:'\"/|_@#$%&+-=<>()[]{}"
OTHERS = '¿'
alphabet = alphabet + OTHERS + ' '

for sent in data:
    for c in sent:
        vocab[c] +=1
UNK_char = set(vocab) - set(alphabet) - {' '}
vocab_new = Counter()
vocab_new[' '] = 0
vocab_new[OTHERS] = 0
for c in alphabet:
    vocab_new[c] = vocab[c]
vocab_new[' '] = vocab[' ']
for c in UNK_char:
    vocab_new[OTHERS] += vocab[c]
for c in UNK_char:
    data = [i.replace(c,OTHERS) for i in data]
vocab_new = sorted(list(vocab_new))
data, test_data = train_test_split(data,test_size = 500)
data = ''.join(data)
vocab = vocab_new
char2index = {k:v for v,k in enumerate(vocab_new)}
index2char = {v:k for v,k in enumerate(vocab_new)}

vocab_size = len(vocab_new)

# cut the text in semi-redundant sequences of maxlen characters
maxlen = args.maxlen
sentences = []
next_chars = []
step = 1
for i in range(0, len(data) - maxlen, step):
    sentences.append(data[i: i + maxlen])
    next_chars.append(data[i + maxlen])
print('nb sequences:', len(sentences))

def one_hot_encode_label(label, vocab):
    vec = [1.0 if l==label else 0.0 for l in vocab]
    return vec


def batch_generator_lm(sentences, next_chars,char2index, batch_size=1):
    while True:
        batch_x = []
        batch_y = []
        for sent,c in zip(sentences,next_chars):
            x = np.zeros((maxlen, vocab_size))
            for t, char in enumerate(sent):
                x[t, char2index[char]] = 1
            batch_x.append(x)
            batch_y.append(one_hot_encode_label(c, char2index))
            if len(batch_x) >= batch_size:
                yield np.array(batch_x), np.array(batch_y)
                batch_x = []
                batch_y = []
            # if len(batch_x) >= batch_size:
            #     # Pad Sequences in batch to same length
            #     batch_x = pad_sequences(batch_x, vocab[PAD])
            #     batch_y = pad_sequences(batch_y, one_hot_encode_label(PAD, vocab))
            #     yield np.array(batch_x), np.array(batch_y)
                


# print('Vectorization...')
# x = np.zeros((len(sentences), maxlen, vocab_size), dtype=np.bool)
# y = np.zeros((len(sentences), vocab_size), dtype=np.bool)
# for i, sentence in enumerate(sentences):
    for t, char in enumerate(sent):
        x[i, t, char2index[char]] = 1
#     y[i, char2index[next_chars[i]]] = 1



def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)

index2char = {v:k for v,k in enumerate(vocab_new)}

def on_epoch_end(epoch, _):
    # Function invoked at end of each epoch. Prints generated text.
    print()
    print('----- Generating text after Epoch: %d' % epoch)

    start_index = random.randint(0, len(data) - maxlen - 1)
    generated = ''
    sentence = data[start_index: start_index + maxlen]
    generated += sentence
    print('----- Generating with seed: "' + sentence + '"')
    for i in range(400):
            x_pred = np.zeros((1, maxlen, vocab_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, char2index[char]] = 1.

            preds = model.predict_classes(x_pred, verbose=0)
            
            next_char = index2char[preds[0]]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
    
    
    for diversity in [0.2, 0.5, 1.0, 1.2]:
        print('----- diversity:', diversity)

        generated = ''
        sentence = data[start_index: start_index + maxlen]
        generated += sentence
        print('----- Generating with seed: "' + sentence + '"')
        sys.stdout.write(generated)

        for i in range(400):
            x_pred = np.zeros((1, maxlen, vocab_size))
            for t, char in enumerate(sentence):
                x_pred[0, t, char2index[char]] = 1.

            preds = model.predict(x_pred, verbose=0)[0]
            next_index = sample(preds, diversity)
            next_char = index2char[next_index]

            sentence = sentence[1:] + next_char

            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()

    
weights = ModelCheckpoint(filepath = 'model.h5')
print_callback = LambdaCallback(on_epoch_end=on_epoch_end)

# model.fit(x, y,
#           batch_size=128,
#           epochs=60,
#           callbacks=[print_callback])
# model.fit_generator(batch_generator_lm(sentences, next_chars,char2index,30),
#                              epochs=60, steps_per_epoch=len(sentences)/30,callbacks=[print_callback,weights])


print('Build model...')
model = Sequential()
model.add(LSTM(args.hidden_size, input_shape=(maxlen, vocab_size),return_sequences=True))
model.add(Dropout(args.dropout))
model.add(LSTM(args.hidden_size, return_sequences=True))
model.add(Dropout(args.dropout))
model.add(LSTM(args.hidden_size, return_sequences=True))
model.add(Dropout(args.dropout))
model.add(LSTM(args.hidden_size))
model.add(Dropout(args.dropout))
model.add(Dense(vocab_size, activation='softmax'))

optimizer = RMSprop(learning_rate=args.learning_rate)
model.compile(loss='categorical_crossentropy', optimizer=optimizer)


if not os.path.exists('model.h5'):
    model.fit_generator(batch_generator_lm(sentences, next_chars,char2index,args.batch_size),
                             epochs=args.epochs, steps_per_epoch=len(sentences)/args.batch_size,callbacks=[print_callback,weights])
else:
    model = load_model('model.h5')
    print(model.summary())
    model.fit_generator(batch_generator_lm(sentences, next_chars,char2index,args.batch_size),
                             epochs=args.epochs, steps_per_epoch=len(sentences)/args.batch_size,callbacks=[print_callback,weights])
