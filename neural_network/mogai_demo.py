import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import LSTM, TimeDistributed, Dense
from keras.optimizers import Adam

data = np.load('input.npy')

BATCH_START = 0
TIME_STEPS = 2
BATCH_SIZE = 6
INPUT_SIZE = 5
OUTPUT_SIZE = 2
CELL_SIZE = 20
LR = 0.006

# 数据长度取整
BATCH_NUM = int(len(data / BATCH_SIZE))
data = data[0:BATCH_NUM * BATCH_SIZE]

def batch():
    global BATCH_START, TIME_STEPS
    X_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 1:6]
    X_batch = X_batch.reshape(BATCH_SIZE, TIME_STEPS,5)
    Y_batch = data[BATCH_START: BATCH_START + BATCH_SIZE * TIME_STEPS, 6:8]
    print(Y_batch)
    print(Y_batch.shape)
    Y_batch = Y_batch.reshape(BATCH_SIZE, TIME_STEPS,2)
    BATCH_START += TIME_STEPS
    return [X_batch, Y_batch]

model = Sequential()
# build a LSTM RNN
model.add(LSTM(
    batch_input_shape=(BATCH_SIZE, TIME_STEPS, INPUT_SIZE),       # Or: input_dim=INPUT_SIZE, input_length=TIME_STEPS,
    output_dim=CELL_SIZE,
    return_sequences=True,      # True: output at all steps. False: output as last step.
    stateful=True,              # True: the final state of batch1 is feed into the initial state of batch2
))
# add output layer
model.add(TimeDistributed(Dense(OUTPUT_SIZE)))
adam = Adam(LR)
model.compile(optimizer=adam,
              loss='mse',)


print('Training ------------')
for step in range(BATCH_NUM):
    # data shape = (batch_num, steps, inputs/outputs)
    X_batch, Y_batch = batch()
    cost = model.train_on_batch(X_batch, Y_batch)
    pred = model.predict(X_batch, BATCH_SIZE)
    if step % 10 == 0:
        print('train cost: ', cost)


