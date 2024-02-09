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
from keras.callbacks import ModelCheckpoint, Callback
import matplotlib.pyplot as plt
from scipy.interpolate import interp2d
#from ann_visualizer.visualize import ann_viz;
from PIL import Image, ImageFilter

class SaveWeightsToArray(Callback):
    def __init__(self):
        super(SaveWeightsToArray, self).__init__()
        self.weights_array = []

    def on_epoch_end(self, epoch, logs=None):
        weights = self.model.get_weights()
        weight_matrices = [w for w in weights if len(w.shape) == 2]  # Extract only weight matrices
        self.weights_array.append(weight_matrices)

def get_shape(lst):
    if isinstance(lst, list):
        return [len(lst)] + get_shape(lst[0])
    else:
        return []
    
def remove_directory(directory):

    # Iterate over the directory and delete all files and subdirectories
    for root, dirs, files in os.walk(directory, topdown=False):
        for file in files:
            file_path = os.path.join(root, file)
            os.remove(file_path)
        for dir in dirs:
            dir_path = os.path.join(root, dir)
            os.rmdir(dir_path)

    # Finally, remove the top-level directory
    os.rmdir(directory)

    
def create():
    dataset = pd.read_csv('data/train.csv')
    dataset.head(10)

    x = dataset.iloc[:,:20].values
    y = dataset.iloc[:,20:21].values

    x = StandardScaler().fit_transform(x)
    y = OneHotEncoder().fit_transform(y).toarray()

    x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.1)

    model = Sequential()
    model.add(Dense(18, input_dim=20, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(4, activation='softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    save_weights_callback = SaveWeightsToArray()

    history = model.fit(x_train, y_train, validation_data = (x_test,y_test), epochs=100, batch_size=64,  callbacks=[save_weights_callback])

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

    return save_weights_callback.weights_array

def analyse(array):
    ret = []
    for i in range(len(array)):
        individual_weights = []
        for i, layer in enumerate(array[i]):
            weights = np.array(layer)
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

        ret.append(normalised_array_scaled)
    return ret

def process(array, filename, savename, savename2):
    image = Image.fromarray(array.astype(np.uint8))
    image.save(filename)

    img = cv2.imread(filename)
    res = cv2.resize(img, dsize=(640, 400), interpolation=cv2.INTER_CUBIC)
    res2 = cv2.resize(img, dsize=(640, 400), interpolation=cv2.INTER_NEAREST)
    cv2.imwrite(savename, res)
    cv2.imwrite(savename2, res2)

def video(image_dir):
    # Define the video codec and output file
    codec = cv2.VideoWriter_fourcc(*'XVID')
    output_file = image_dir + '.avi'

    # Get the list of image filenames
    image_files = [os.path.join(image_dir, f) for f in os.listdir(image_dir) if f.endswith('.png')]

    # Get the dimensions of the first image
    img = cv2.imread(image_files[0])
    height, width, _ = img.shape

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_file, codec, 30.0, (width, height))

    # Write each image to the video file
    for image_file in image_files:
        img = cv2.imread(image_file)
        out.write(img)

    # Release the VideoWriter
    out.release()

    print("Video created successfully!")


os.makedirs('smooth')
os.makedirs('unfiltered')
os.makedirs('expanded')

weights = create()
weights = analyse(weights)
for i in range(len(weights)):
    filename = f"unfiltered/unfiltered_b&w_{i:03d}.png"
    savename = f"smooth/smooth_b&w_{i:03d}.png"
    savename2 = f"expanded/expanded_b&w_{i:03d}.png"
    process(weights[i], filename, savename, savename2)
video('smooth')
#video('expanded')

remove_directory('smooth')
remove_directory('weights')
remove_directory('unfiltered')
remove_directory('expanded')
