import numpy as np
import pandas as pd
import cv2
import os
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.callbacks import ModelCheckpoint
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
#from ann_visualizer.visualize import ann_viz;
from PIL import Image, ImageFilter
import random
import noise

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

    # Create a folder to save the weights
    # weights_folder = 'weights'
    # os.makedirs(weights_folder, exist_ok=True)

    # # Define the filepath for saving weights
    # weights_filepath = os.path.join(weights_folder, 'weights_epoch_{epoch:02d}.keras')



    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = model.fit(x_train, y_train, validation_data = (x_test,y_test), epochs=100, batch_size=64)

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
    print("model loaded")

    individual_weights = []
    for i, layer in enumerate(model.layers):
        weights = np.array(layer.get_weights()[0])
        individual_weights.append(weights)
    print('weights acquired')
    
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
    #image.show()

def red_green(array):
    normalized_array = array / 255.0

    # Create a Pillow Image using the red to green gradient
    colored_array = np.stack([1 - normalized_array, normalized_array, np.zeros_like(normalized_array)], axis=-1)
    
    # Convert to integers and create a Pillow Image
    image = Image.fromarray((colored_array * 255).astype(np.uint8), mode='RGB')
    image.save('output_r&g.png')

def resize():
    # Assuming you have an image named 'original_image'
    img = Image.open("output_r&g.png")
    original_array = np.array(img)
    
    # Define the desired size and the number of points for interpolation
    new_size = (400, 640)
    interpolation_factor = 10  # Increase for more points, decrease for fewer points

    # Calculate scaling factors
    scale_x = img.width / new_size[1]
    scale_y = img.height / new_size[0]

    # Initialize new image
    resized_img = Image.new("RGB", (new_size[1], new_size[0]))

    #size = 1 for cool blocky look
    neighborhood_size = 2
    # Half of neighborhood size
    n_half = neighborhood_size // 2

    # Iterate through each pixel in the new image
    for i in range(new_size[0]):
        for j in range(new_size[1]):
            # Calculate corresponding position in original image
            x_orig = j * scale_x
            y_orig = i * scale_y

            # Nearest neighbor interpolation
            x_low = max(int(x_orig) - n_half, 0)
            y_low = max(int(y_orig) - n_half, 0)
            x_high = min(x_low + neighborhood_size, img.width)
            y_high = min(y_low + neighborhood_size, img.height)

            x_weight = x_orig - x_low
            y_weight = y_orig - y_low

            # Get pixel values of surrounding pixels
            pixels = [img.getpixel((x, y)) for y in range(y_low, y_high) for x in range(x_low, x_high)]

            # Perform bilinear interpolation for each channel
            interpolated_value = tuple(
                int(sum((1 - x_weight) * (1 - y_weight) * pixel[c] for pixel in pixels))
                for c in range(3)  # 3 channels (RGB)
            )

            # Assign interpolated pixel value
            resized_img.putpixel((j, i), interpolated_value)


    resized_img.show()
    return resized_img

def cv2resize():
    img = cv2.imread('output_b&w.png')
    res = cv2.resize(img, dsize=(1920, 1080), interpolation=cv2.INTER_NEAREST)
    # smoothed_image = cv2.GaussianBlur(res, (1, 1), 0)
    # cv2.imshow("smoothed image", cv2.GaussianBlur(res, (5, 5), 0))
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    # Define the dimensions of each segment
    segment_width = 1920 // 38
    segment_height = 1080 // 20

    # Define a function to move the borders
    def move_borders(res):
        # Create a copy of the image
        new_image = np.copy(res)
        
        # Iterate over vertical borders
        for y in range(1, 20):
            shift_amount = random.randint(-5, 5)
            new_image[y * segment_height : y * segment_height + shift_amount, :] = res[y * segment_height - shift_amount : y * segment_height, :]
        
        # Iterate over horizontal borders
        for x in range(1, 38):
            shift_amount = random.randint(-5, 5)
            new_image[:, x * segment_width : x * segment_width + shift_amount] = res[:, x * segment_width - shift_amount : x * segment_width]

        return new_image

    # Function to continuously update and display the image
    def update_image():
        while True:
            moved_image = move_borders(res)
            cv2.imshow('Fluid Grid', moved_image)
            if cv2.waitKey(100) & 0xFF == ord('q'):  # Press 'q' to quit
                break

    # Call the function to start displaying the image
    update_image()

    # Release resources
    cv2.destroyAllWindows()


def resize_blur(array):
    pixel_width = array.shape[1]
    pixel_height = array.shape[0]
    resize_dims = (640, 400)

    upsample_factor_x = resize_dims[0] // pixel_width
    upsample_factor_y = resize_dims[1]  // pixel_height

    expanded_array = np.kron(array, np.ones((upsample_factor_y, upsample_factor_x)))
    expanded_array = expanded_array[:400, :640]

    blur_radius = upsample_factor_x // 3

    image = Image.fromarray(expanded_array.astype(np.uint8))
    smoothed_image = image.filter(ImageFilter.GaussianBlur(blur_radius))
    smoothed_image.save("b&w_expanded_blur.png")


#create()
weights = analyse()
cv2resize()
#red_green(weights)
#weights = resize()

