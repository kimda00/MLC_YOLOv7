import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import os
import shutil
import tensorflow as tf

from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPool2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

def load_data(label_path, image_dir):
    # Load data from CSV file
    data = pd.read_csv(label_path)
    print(data.shape)
    print(data.head())

    img_width = 150
    img_height = 150

    X = []

    # Load and preprocess images
    for i in tqdm(range(data.shape[0])):
        path = os.path.join('inference/pick_1000', data['Id'][i] + '.jpg')
        img = image.load_img(path, target_size=(img_width, img_height, 3))
        img = image.img_to_array(img)
        img = img / 255.0
        X.append(img)

    X = np.array(X)
    # print(X.shape)
    y = data.drop(['Id', 'State'], axis=1)
    y = y.to_numpy()
    # print(y.shape)

    return X, y

def gpu_check():
    # GPU 장치 목록 확인
    physical_devices = tf.config.list_physical_devices('GPU')
    if len(physical_devices) > 0:
        print("사용 가능한 GPU 장치 목록:")
        for device in physical_devices:
            print(device)
    else:
        print("사용 가능한 GPU 장치가 없습니다.")


def build_model(input_shape, num_classes):

    # Build the model
    model = Sequential([
        Conv2D(16, (3,3), activation='relu', input_shape=input_shape),
        MaxPool2D(2, 2),
        Conv2D(32, (3,3), activation='relu'),
        MaxPool2D(2, 2),
        Conv2D(64, (3,3), activation='relu'),
        MaxPool2D(2, 2),
        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(y_train.shape[1], activation='sigmoid')
    ])

    print(model.summary())
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, X_train, y_train, epochs=32, batch_size=16, X_test=None, y_test=None):
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, y_test))
    return history

def save_model(model, save_directory, model_weights_filename):
    os.makedirs(save_directory, exist_ok=True)
    model.save(os.path.join(save_directory, model_weights_filename))
    print(f"Model weights saved as '{model_weights_filename}'")

label_path = 'inference/1000_selected_images.csv'
image_dir = 'inference/pick_1000'
model_weights_filename = 'pick_1000_E32.h5'


X, y = load_data(label_path, image_dir)
gpu_check()

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0, test_size=0.15)
# print(X_train)
model = build_model(X_train[0].shape, y_train.shape[1])
history = train_model(model, X_train, y_train, epochs=32, batch_size=32, X_test=X_test, y_test=y_test)
save_model(model, 'weights', model_weights_filename)


# # # Plot training & validation accuracy values
# # epoch_range = range(1, epochs+1)
# # plt.plot(epoch_range, history.history['accuracy'])
# # plt.plot(epoch_range, history.history['val_accuracy'])
# # plt.title('Model accuracy')
# # plt.ylabel('Accuracy')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Val'], loc='upper left')
# # plt.show()

# # # Plot training & validation loss values
# # plt.plot(epoch_range, history.history['loss'])
# # plt.plot(epoch_range, history.history['val_loss'])
# # plt.title('Model loss')
# # plt.ylabel('Loss')
# # plt.xlabel('Epoch')
# # plt.legend(['Train', 'Val'], loc='upper left')
# # plt.show()

# # # Plot learning curve
# # # plot_learningCurve(history, epochs)

