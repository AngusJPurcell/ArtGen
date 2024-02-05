import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
#from ann_visualizer.visualize import ann_viz;
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

    individual_weights = []
    for i, layer in enumerate(model.layers):
        weights = np.array(layer.get_weights()[0])
        individual_weights.append(weights)
    
    min_value = np.min(individual_weights[0])
    max_value = np.max(individual_weights[0])

    # Iterate through the remaining arrays in the list
    for arr in individual_weights[1:]:
        min_value = min(min_value, np.min(arr))
        max_value = max(max_value, np.max(arr))

    for i in range(len(individual_weights)):
        individual_weights[i] += (min_value * (-1))

    max_value += (min_value * (-1))
    min_value += (min_value * (-1))
        
    max_rows = max(individual_weights[0].shape[0], individual_weights[1].shape[0], individual_weights[2].shape[0])
    max_cols = max(individual_weights[0].shape[1], individual_weights[1].shape[1], individual_weights[2].shape[1])

    pad_top_arr1 = (max_rows - individual_weights[0].shape[0]) // 2
    pad_bottom_arr1 = max_rows - individual_weights[0].shape[0] - pad_top_arr1

    pad_top_arr2 = (max_rows - individual_weights[1].shape[0]) // 2
    pad_bottom_arr2 = max_rows - individual_weights[1].shape[0] - pad_top_arr2

    pad_top_arr3 = (max_rows - individual_weights[2].shape[0]) // 2
    pad_bottom_arr3 = max_rows - individual_weights[2].shape[0] - pad_top_arr3

    result = np.concatenate([
        np.pad(individual_weights[0], ((pad_top_arr1, pad_bottom_arr1), (0, 0))),
        np.pad(individual_weights[1], ((pad_top_arr2, pad_bottom_arr2), (0, 0))),
        np.pad(individual_weights[2], ((pad_top_arr3, pad_bottom_arr3), (0, 0)))
    ], axis=1)

    normalised_array = (result - min_value) / (max_value - min_value)
    normalised_array_scaled = normalised_array * 255
    return normalised_array_scaled

def black_white(array):
    # Create a PIL Image from the 3D array
    image = Image.fromarray(array.astype(np.uint8))
    image.save('output_b&w.png')
    image.show()

def create_blue_gradient(value):
    return (0, 0, value)

def red_green(array):
    normalized_array = array / 255.0
    print(normalized_array)

    # Create a Pillow Image using the red to green gradient
    colored_array = np.stack([1 - normalized_array, normalized_array, np.zeros_like(normalized_array)], axis=-1)
    
    # Convert to integers and create a Pillow Image
    image = Image.fromarray((colored_array * 255).astype(np.uint8), mode='RGB')
    image.save('output_r&g.png')
    image.show()

def resize(array):
    original_size = array.shape
    desired_size = (1080, 1728)

    # Create coordinate grids for original and desired sizes
    x_orig, y_orig = np.linspace(0, 1, original_size[1]), np.linspace(0, 1, original_size[0])
    x_new, y_new = np.linspace(0, 1, desired_size[1]), np.linspace(0, 1, desired_size[0])

    # Create interpolation function
    #interp_func = interp2d(x_orig, y_orig, array, kind='linear')
    interp_func = interp2d(x_orig, y_orig, array, kind='nearest')

    # Interpolate values for the new size
    interpolated_array = interp_func(x_new, y_new)

    # Display the original and interpolated arrays
    # plt.imshow(array, cmap='viridis', extent=[0, 1, 0, 1])
    # plt.title('Original Array')
    # plt.show()

    # plt.imshow(interpolated_array, cmap='viridis', extent=[0, 1, 0, 1])
    # plt.title('Interpolated Array')
    # plt.show()
    return interpolated_array

weights = analyse()
weights = resize(weights)

#black_white(weights)
red_green(weights)
