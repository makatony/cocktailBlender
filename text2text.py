import json
import numpy as np
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import LSTM, RepeatVector, Dense, Dropout, Activation
from keras.layers.wrappers import TimeDistributed

drinks = json.load(open('cocktaildb/text2text_training_data.json'))


chars = set()
for k in drinks.keys():
    for ch in k:
        chars.add(ch)
    for ch in drinks[k]:
        chars.add(ch)
CHARS = list(chars) + ['<EOS>']
CHAR_TO_INDEX = {i: c for c, i in enumerate(CHARS)}
INDEX_TO_CHAR = {c: i for c, i in enumerate(CHARS)}


def one_hot_to_index(vector):
    if not np.any(vector):
        return -1

    return np.argmax(vector)

def one_hot_to_char(vector):
    index = one_hot_to_index(vector)
    if index == -1:
        return ''

    return INDEX_TO_CHAR[index]

def one_hot_to_string(matrix):
    return ''.join(one_hot_to_char(vector) for vector in matrix)

max_len_x = max([len(k) for k in drinks.keys()]) + 1
max_len_y = max([len(drinks[k]) for k in drinks.keys()]) + 1

def prepare_data():
    x = np.zeros((len(drinks), max_len_x, len(CHARS)), dtype=np.bool)
    y = np.zeros((len(drinks), max_len_y, len(CHARS)), dtype=np.bool)
    for i, drink_name in enumerate(drinks):
        drink_instructions = drinks[k]
        for t, char in enumerate(drink_name):
            x[i, t, CHAR_TO_INDEX[char]] = 1
        x[i, len(drink_name), CHAR_TO_INDEX['<EOS>']] = 1
        for t, char in enumerate(drink_instructions):
            y[i, t, CHAR_TO_INDEX[char]] = 1
        y[i, len(drink_instructions), CHAR_TO_INDEX['<EOS>']] = 1
    return train_test_split(x, y, test_size=0.066666)

train_X, test_X, train_y, test_y = prepare_data()

model = Sequential()
model.add(LSTM(256, input_shape=(max_len_x, len(CHARS))))
model.add(Dropout(0.25))
model.add(RepeatVector(max_len_y))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.25))
model.add(TimeDistributed(Dense(len(chars) + 1)))
model.add(Activation('softmax'))

model.compile(
    loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])

model.fit(
    train_X, train_y, epochs=1000, batch_size=512, validation_data=(test_X, test_y))

num_preds = 3
pred_indices = np.random.choice(
    test_X.shape[0], size=num_preds, replace=False)
predictions = model.predict(test_X[pred_indices, :])

for i in range(num_preds):
    print('\n{}\n{}\n'.format(
        one_hot_to_string(test_X[pred_indices[i]]),
        one_hot_to_string(predictions[i])))
