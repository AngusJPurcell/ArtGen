import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from ann_visualizer.visualize import ann_viz;
from PIL import Image

def create():
    dataset = pd.read_csv('data/train.csv')
    dataset.head(10)

    x = dataset.iloc[:,:20].values
    y = dataset.iloc[:,20:21].values

    x = StandardScaler().fit_transform(x)
    y = OneHotEncoder().fit_transform(y).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1)

    model = Sequential()
    model.add(Dense(16, input_dim=20, activation='relu'))
    model.add(Dense(12, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train,validation_data = (x_test,y_test), epochs=100, batch_size=64)

    model.save("ann_100epoch.keras")
    # plt.plot(history.history['accuracy'])
    # plt.plot(history.history['val_accuracy'])
    # plt.title('Model accuracy')
    # plt.ylabel('Accuracy')
    # plt.xlabel('Epoch')
    # plt.legend(['Train', 'Test'], loc='upper left')
    # plt.show()

    # plt.plot(history.history['loss'])
    # plt.plot(history.history['val_loss']) 
    # plt.title('Model loss') 
    # plt.ylabel('Loss') 
    # plt.xlabel('Epoch') 
    # plt.legend(['Train', 'Test'], loc='upper left') 
    # plt.show()

    #ann_viz(model, title="ANN 100 epoch")

    return

def analyse():
    model = keras.saving.load_model("ann_100epoch.keras")

    scaled_weights_list = []
    for layer in model.layers:
        weights = np.array(layer.get_weights()[0])
        weights = ((weights + 1) * 127.5).astype(np.uint8)

        for i in range(len(weights)):
            print(weights[i])
            scaled_weights_list.append(weights[i])
        # print(weights.shape, weights)
        # print(scaled_weights.shape, scaled_weights)
            
    scaled_weights = np.array(scaled_weights_list)
    print(scaled_weights.shape, scaled_weights)
    
    image = Image.fromarray(scaled_weights)
    image.save('output.png')
    image.show()


analyse()
