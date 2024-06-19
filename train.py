import os
import librosa
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
import shutil
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt


DATASET_PATH = 'DATASET'
GENRES = 'jazz pop rock'.split()
SAMPLES_TO_CONSIDER = 22050 * 30  # 30 секунд аудио

def load_data(dataset_path, genres):
    data = []
    labels = []
    for genre in genres:
        genre_path = os.path.join(dataset_path, genre)
        for file_name in os.listdir(genre_path):
            if file_name.endswith(".wav"):
                file_path = os.path.join(genre_path, file_name)
                y, sr = librosa.load(file_path, sr=22050)
                if len(y) >= SAMPLES_TO_CONSIDER:
                    y = y[:SAMPLES_TO_CONSIDER]
                    data.append(y)
                    labels.append(genre)
    return np.array(data), np.array(labels)

data, labels = load_data(DATASET_PATH, GENRES)
def extract_features(y, sr):
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean

features = np.array([extract_features(y, 22050) for y in data])
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

model_path = 'music_genre_classification_model.h5'

if not os.path.exists(model_path):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.5),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(len(GENRES), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    model.summary()

    history = model.fit(X_train, y_train, epochs=500, batch_size=32, validation_data=(X_test, y_test))
    model.save(model_path)

    plt.plot(history.history['loss'], label='Train loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

else:
    model = load_model(model_path)

loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")

THRESHOLD = 0.5  # Порог для классификации в жанр, если ниже - в "Прочее"

def classify_and_move_files(model, input_folder, output_folder, label_encoder, threshold):
    for file_name in os.listdir(input_folder):
        if file_name.endswith(".mp3"):
            file_path = os.path.join(input_folder, file_name)
            y, sr = librosa.load(file_path, sr=22050)
            if len(y) >= SAMPLES_TO_CONSIDER:
                y = y[:SAMPLES_TO_CONSIDER]
                features = extract_features(y, sr).reshape(1, -1)
                predictions = model.predict(features)
                genre_index = predictions.argmax(axis=1)[0]
                genre_probability = predictions[0][genre_index]
                
                if genre_probability >= threshold:
                    genre = label_encoder.inverse_transform([genre_index])[0]
                else:
                    genre = 'Other'
                
                genre_folder = os.path.join(output_folder, genre)
                os.makedirs(genre_folder, exist_ok=True)
                shutil.move(file_path, os.path.join(genre_folder, file_name))

classify_and_move_files(model, 'input', 'output', label_encoder, THRESHOLD)
# jazz0054 Ошибка