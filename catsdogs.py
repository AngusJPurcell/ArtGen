import os
import numpy as np
import keras
from keras import layers
from tensorflow import data as tf_data
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

# num_skipped = 0
# for folder_name in ("Cat", "Dog"):
    # folder_path = os.path.join("PetImages", folder_name)
    # for fname in os.listdir(folder_path):
        # fpath = os.path.join(folder_path, fname)
        # try:
            # fobj = open(fpath, "rb")
            # is_jfif = b"JFIF" in fobj.peek(10)
        # finally:
            # fobj.close()
# 
        # if not is_jfif:
            # num_skipped += 1
            # os.remove(fpath)
# 
# print(f"Deleted {num_skipped} images.")
    
image_size = (180, 180)
batch_size = 128

train_ds, val_ds = keras.utils.image_dataset_from_directory(
    "PetImages",
    validation_split=0.2,
    subset="both",
    seed=1337,
    image_size=image_size,
    batch_size=batch_size,
)

img = mpimg.imread('PetImages/Cat/554.jpg')
imgplot = plt.imshow(img)
plt.show()
# 
plt.figure(figsize=(10, 10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i + 1)
        plt.imshow(np.array(images[i]).astype("uint8"))#
        plt.title(int(labels[i]))
        plt.axis("off")
plt.show()

# img = mpimg.imread('PetImages/Cat/1.jpg')
# imgplot = plt.imshow(img)
# plt.show()
# show = True
# for folder_name in ("Cat", "Dog"):
    # folder_path = os.path.join("PetImages", folder_name)
    # for fname in os.listdir(folder_path):
        # fpath = os.path.join(folder_path, fname)
        # try:
            # img = mpimg.imread(fpath)
            # imgplot = plt.imshow(img)
            # if (show == True):
                # print(show)
                # plt.show()
                # show = False
        # except:
            # print(fpath)