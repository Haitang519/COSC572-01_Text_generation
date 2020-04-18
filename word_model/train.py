import os
import numpy as np
from keras.utils import to_categorical
from keras.initializers import Constant
from keras.layers import Embedding, LSTM, Dense, TimeDistributed
from keras.models import Sequential

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class WordModel:

    def __init__(self, length, word_index):
        self.model = Sequential()
        self.word_index = word_index
        self.sentence_length = length
        self.embedding_size = 100
        self.hidden_size = 10
        self.dropout = 0.3
        self.epochs = 3
        self.batch_size = 10

    def load_pretrained_embeddings(self, glove_file):
        embeddings_index = {}
        embedding_matrix = np.zeros((len(self.word_index), self.embedding_size))
        with open(glove_file, encoding='utf8') as f:
            for line in f:
                word, coefs = line.split(maxsplit=1)
                coefs = np.fromstring(coefs, 'f', sep=' ')
                embeddings_index[word] = coefs
            print('Found %s word vectors.' % len(embeddings_index))

            for word, i in self.word_index.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i - 1] = embedding_vector

        return embedding_matrix

    def shift_by_one(self, sent):
        sent = np.delete(sent, 0)
        sent = np.append(sent, 0)
        return sent

    def batch_generator_lm(self, data):
        while True:
            batch_x = []
            batch_y = []
            for sent in data:
                batch_x.append(sent)
                batch_y.append(
                    [to_categorical(token, num_classes=(len(self.word_index))) for token in self.shift_by_one(sent)])
                if len(batch_x) >= self.batch_size:
                    yield np.array(batch_x), np.array(batch_y)
                    batch_x = []
                    batch_y = []

    def describe_data(self, generator):
        batch_x, batch_y = [], []
        for bx, by in generator:
            batch_x = bx
            batch_y = by
            break
        print('Batch input shape:', batch_x.shape)
        print('Batch output shape:', batch_y.shape)

    def build(self):
        embedding_matrix = self.load_pretrained_embeddings('glove.6B/glove.6B.100d.txt')
        self.model.add(Embedding(len(word_index), self.embedding_size, input_length=self.sentence_length,
                                 embeddings_initializer=Constant(embedding_matrix)))
        self.model.add(LSTM(self.hidden_size, dropout=self.dropout, return_sequences=True))
        self.model.add(
            TimeDistributed(Dense(121109, activation='softmax', input_dim=len(self.word_index), use_bias=True),
                            input_shape=(self.batch_size, self.sentence_length)))

        print(self.model.summary())

    def train(self):
        self.model.compile(optimizer='adadelta', loss='categorical_crossentropy', metrics=['accuracy'])

        # Training
        self.model.fit_generator(self.batch_generator_lm(train_data),
                                 epochs=self.epochs, steps_per_epoch=len(train_data) / self.batch_size)

        # Evaluation
        loss, acc = self.model.evaluate_generator(self.batch_generator_lm(val_data), steps=len(val_data))
        print('Dev Loss:', loss, 'Dev Acc:', acc)


def load_data():
    data = np.load('save/data.npz')

    f = open('save/word_index.txt', 'r')
    word_dict = eval(f.read())
    f.close()

    return data['train'], data['val'], int(data['length']), word_dict


if __name__ == "__main__":
    train_data, val_data, length, word_index = load_data()
    word_model = WordModel(length, word_index)
    word_model.describe_data(word_model.batch_generator_lm(train_data))
    word_model.build()
    word_model.train()
