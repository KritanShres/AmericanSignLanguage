import os
import matplotlib.pyplot as plt
import matplotlib.image as mimg
import pandas as pd
import streamlit as st
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import math
import time
from PIL import Image

main_folder_path = r"D:\SixthSemProjects\MinorProject\ASL_Alphabet_Dataset\asl_alphabet_train"
subfolders = [folder for folder in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, folder))]

categories = {}
label = []
count = []


st.header("Dataset Description")
st.subheader("About the Dataset")
st.write("This dataset contains images of alphabets from the American Sign Language (ASL), categorized into 29 different folders, each representing a unique class.")
st.subheader("Context")
st.write("Communication barriers exist between sign language users and non-sign language users due to differences in primary modes of interaction. This dataset aims to facilitate research and applications that bridge this communication gap by providing a structured dataset for ASL recognition.")
st.write('This dataset was compiled by pooling multiple open-source datasets to create a comprehensive collection for ASL recognition research.The dataset is intended to support projects that help reduce the communication gap between sign language users and non-users through machine learning and AI-based applications.')
st.markdown("[View Dataset on Kaggle](https://www.kaggle.com/datasets/debashishsau/aslamerican-sign-language-aplhabet-dataset)")

st.header("Dataset Preview")

rows, columns = 6, 5
fig, axs = plt.subplots(rows, columns, figsize=(10, 10))
ax = axs.ravel()
for index, subfolder in enumerate(sorted(subfolders)):
    ax[index].set_xticks([])
    ax[index].set_yticks([])
    categories[index] = subfolder

    subfolder_path = os.path.join(main_folder_path, subfolder)
    image_files = os.listdir(subfolder_path)

    count.append(len(image_files))
    label.append(subfolder)

    image_path = os.path.join(subfolder_path, image_files[0])
    img = mimg.imread(image_path)

    ax[index].imshow(img)
    ax[index].set_title(subfolder)
    ax[index].axis('off')

for i in range(index+1, len(ax)):
    ax[i].axis('off')

st.pyplot(fig)

df = pd.DataFrame({'Label': label, 'Count': count})

st.subheader('Bar graph for Dataset')
image_counts = {}

for letter in subfolders:
    subfolder_path = os.path.join(main_folder_path, letter)
    letter_files = os.listdir(subfolder_path)
    image_counts[letter] = len(letter_files)

sorted_letters = sorted(image_counts.keys())
sorted_counts = [image_counts[letter] for letter in sorted_letters]

fig_bar = plt.figure(figsize=(15, 15))
bars = plt.bar(sorted_letters, sorted_counts, color='yellow')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())),
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.xlabel('Letter')
plt.ylabel('Number of Images')
plt.title('Number of Images per Letter in ASL Dataset')
plt.xticks(rotation=90)
plt.tight_layout()

col1, col2 = st.columns([4, 4])

with col1:
    st.dataframe(df)

with col2:
    st.pyplot(fig_bar)

main_folder_path = r'D:\SixthSemProjects\ASL_Alphabet_Dataset\randomly_selected_train_5k'
subfolders = [folder for folder in os.listdir(main_folder_path) if os.path.isdir(os.path.join(main_folder_path, folder))]

image_counts = {}

for letter in subfolders:
    subfolder_path = os.path.join(main_folder_path, letter)
    letter_files = os.listdir(subfolder_path)
    image_counts[letter] = len(letter_files)

sorted_letters = sorted(image_counts.keys())
sorted_counts = [image_counts[letter] for letter in sorted_letters]

st.subheader("Uniform Dataset with 5000 Images")
st.write("For this 5000 images have been choosen from each dataset, since the author of the dataset has not mentioned the reason for inconsistency in the dataset, we choose to create a uniform dataset for training.")
st.write("We choose images with a random seed.")
plt.figure(figsize=(12, 6))
bars = plt.bar(sorted_letters, sorted_counts, color='green')

for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), str(int(bar.get_height())),
             ha='center', va='bottom', fontweight='bold', fontsize=10)

plt.xlabel('Letter')
plt.ylabel('Number of Images')
plt.title('Number of Images per Letter in ASL Dataset')
plt.xticks(rotation=90)
plt.tight_layout()

st.pyplot(plt)

folder_path = r'D:\SixthSemProjects\ASL_Alphabet_Dataset\randomly_selected_train_5k\A'
image_files = os.listdir(folder_path)

random_images = random.sample(image_files, 20)

st.subheader("Dataset Diverisity Letter A")
st.write("Randomly choosen 20 images from our dataset. Diverse dataset is useful for training validation and to avoid overfitting as well. During each epoch of training, the model ajusts its hyperparameters such as learning rate and dropout rate based on the model comparision to validation dataset.")
st.write("So, choosen dataset could provide such advantage")
fig, axs = plt.subplots(4, 5, figsize=(15, 10))
ax = axs.ravel()

for i, img_file in enumerate(random_images):
    img_path = os.path.join(folder_path, img_file)
    img = mimg.imread(img_path)
    ax[i].imshow(img)
    ax[i].axis('off')
    ax[i].set_title(f"Image {i+1}")

plt.tight_layout()
st.pyplot(plt)

st.subheader("Dataset augmentation result (if was needed)")
st.write("If the dataset was not small enough to be trained on a custom model (not a pretrained model), data augmentation would've been necessary. From the above sample of letter 'a' images, shows the diveristy needed to train on a custom CNN model    ")

sample_img = tf.keras.preprocessing.image.load_img(r"D:\SixthSemProjects\ASL_Alphabet_Dataset\asl_alphabet_train\A\1.jpg")
sample_img = tf.keras.preprocessing.image.img_to_array(sample_img)
sample_img = np.expand_dims(sample_img, axis=0)

datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    brightness_range=[0.7, 1.3]
)

augmented_images = [datagen.flow(sample_img, batch_size=1).__next__()[0].astype('uint8') for _ in range(6)]
augmentations = ['Rotation', 'Width Shift', 'Height Shift', 'Zoom Range', 'Horizontal Flip', 'Brightness Range']

fig, axes = plt.subplots(1, 7, figsize=(15, 5))
axes[0].imshow(sample_img[0].astype('uint8'))
axes[0].set_title("Original")

for i, img in enumerate(augmented_images):
    axes[i+1].imshow(img)
    axes[i+1].set_title(augmentations[i])

plt.tight_layout()
st.pyplot(plt)


#original image vs resized image
st.header("Original Image vs resized image")
st.write("The convolutional filter will be applied to the image size of 128x128, so any images of variable sizes within the dataset will be resized to a standard Medium sized image. For image classification type Deep Learning model, image sizes of 96x96 to 128x128 are considered medium sized. And images from 224x224 to 256x256 are considered large images. To save computation but still to retain the spatial features of the image, we choose the highest resolution of medium image ie 128x128.")
image = cv2.imread(r"D:\SixthSemProjects\ASL_Alphabet_Dataset\asl_alphabet_train\A\1.jpg")

if image is None:
    st.error("Error: Image not loaded. Please check the file path.")
else:
    resized_image = cv2.resize(image, (128, 128))
    resized_image_2 = cv2.resize(image, (240,240))
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    resized_image_rgb = cv2.cvtColor(resized_image, cv2.COLOR_BGR2RGB)

    st.subheader("Original Image Sample (1920x1920)")
    st.image(image_rgb, caption="Original Image", use_container_width=True)

    st.subheader("Resized Image (128x128)")
    st.image(resized_image_rgb, caption="Resized Image",use_container_width=True)

