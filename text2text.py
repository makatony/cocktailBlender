import json
import numpy as np

drinks = json.load(open('cocktaildb/text2text_training_data.json'))


from recurrentshop import RecurrentSequential
RecurrentSequential._preprocess_function = None

import seq2seq
from seq2seq.models import SimpleSeq2Seq

max_len_x = max([len(k) for k in drinks.keys()])
max_len_y = max([len(drinks[k]) for k in drinks.keys()])

train_X = []
train_Y = []

EOS = 0
PAD = 1

for k in drinks.keys():
    train_X.append([ord(ch) for ch in k] + [EOS] + list([PAD] * (max_len_x - len(k))))
    train_Y.append([ord(ch) for ch in drinks[k]] + [EOS] + list([PAD] * (max_len_y - len(drinks[k]))))
train_X = np.array([train_X])
train_Y = np.array([train_Y])

model = SimpleSeq2Seq(input_dim=max_len_x + 1, hidden_dim=10, output_length=8, output_dim=max_len_y + 1)
model.compile(loss='mse', optimizer='rmsprop')

model.fit(train_X, train_Y, batch_size=64, epochs=1, verbose=2)
