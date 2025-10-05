import tensorflow as tf
from tensorflow import keras


SEQUENCE_LENGTH = 30
NUM_FEATURES = 5


model = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(SEQUENCE_LENGTH, NUM_FEATURES)),
        keras.layers.Dropout(0.2),
        keras.layers.LSTM(32),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

model.save("models/model_rnn_1.keras")