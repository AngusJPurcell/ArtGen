import numpy as np
import pandas as pd
import cv2
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

def resize():
    # Assuming you have an image named 'original_image'
    original_image = Image.open("output_r&g.png")
    original_array = np.array(original_image)
    
    # Define the desired size and the number of points for interpolation
    desired_size = (640, 400)
    interpolation_factor = 10  # Increase for more points, decrease for fewer points

    # Create coordinate grids for original and desired sizes
    x_orig, y_orig = np.linspace(0, 1, original_array.shape[1]), np.linspace(0, 1, original_array.shape[0])
    x_new, y_new = np.linspace(0, 1, desired_size[1]), np.linspace(0, 1, desired_size[0])

    # Perform linear interpolation manually
    interpolated_array = np.zeros((desired_size[0], desired_size[1], original_array.shape[2]), dtype=np.uint8)

    for channel in range(original_array.shape[2]):
        for i in range(desired_size[0]):
            for j in range(desired_size[1]):
                x_idx = int(j * (original_array.shape[1] - 1) / (desired_size[1] - 1))
                y_idx = int(i * (original_array.shape[0] - 1) / (desired_size[0] - 1))

                x_low, x_high = x_orig[max(x_idx - 1, 0)], x_orig[min(x_idx + 1, len(x_orig) - 1)]
                y_low, y_high = y_orig[max(y_idx - 1, 0)], y_orig[min(y_idx + 1, len(y_orig) - 1)]

                interpolated_array[i, j, channel] = (
                    (x_high - x_new[j]) * (y_high - y_new[i]) * original_array[y_low, x_low, channel] +
                    (x_new[j] - x_low) * (y_high - y_new[i]) * original_array[y_low, x_high, channel] +
                    (x_high - x_new[j]) * (y_new[i] - y_low) * original_array[y_high, x_low, channel] +
                    (x_new[j] - x_low) * (y_new[i] - y_low) * original_array[y_high, x_high, channel]
                ).astype(np.uint8)

    # Create a new image from the interpolated array
    interpolated_image = Image.fromarray(interpolated_array)

    # Display the original and interpolated images
    plt.imshow(original_array, cmap='viridis', extent=[0, 1, 0, 1])
    plt.title('Original Image')
    plt.show()

    plt.imshow(interpolated_array, cmap='viridis', extent=[0, 1, 0, 1])
    plt.title(f'Interpolated Image (Manual, {interpolation_factor}x Interpolation)')
    plt.show()

    return interpolated_array

def cv2resize():
    img = cv2.imread('output_b&w.png')
    res = cv2.resize(img, dsize=(640, 400), interpolation=cv2.INTER_)
    cv2.imwrite("black&white.png", res)


weights = analyse()
red_green(weights)
weights = resize()

#black_white(weights)
