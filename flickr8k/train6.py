import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import Sequence, to_categorical
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Add, Attention, BatchNormalization, Conv2D, MaxPooling2D, GlobalAveragePooling2D, Activation, Dropout, Flatten, Dense, Input, Layer, Embedding, LSTM, add, Concatenate, Reshape, concatenate, Bidirectional
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout, Embedding, LSTM, Bidirectional, Attention, Concatenate, Add, MultiHeadAttention, LeakyReLU, Reshape
from tensorflow.keras.applications import VGG16, DenseNet201
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
import json
from textwrap import wrap

# Set display preferences of plot
plt.rcParams['font.size'] = 12
sns.set_style("dark")
warnings.filterwarnings('ignore')

#################################
#       Pre-processing          #
#################################

print("Loading and preparing dataset...")

# Define dataset paths and load caption data
image_path = './dataset/Images'
data = pd.read_csv("./dataset/captions.txt")
data.head()

# Define function to read and process images
def readImage(path, img_size=224):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Convert to array and normalize
    return img

# Function to display a sample of images with their captions
def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    n = 0
    for i in range(15):
        n += 1
        plt.subplot(5, 5, n)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        image = readImage(f"./dataset/Images/{temp_df.image[i]}")
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")

# Display a random sample of 15 images with captions
display_images(data.sample(15))

# Text preprocessing function
def text_preprocessing(data):
    data['caption'] = data['caption'].apply(lambda x: x.lower())  # Lowercase all text
    data['caption'] = data['caption'].apply(lambda x: x.replace("[^A-Za-z]", ""))  # Remove non-alphabetic chars
    data['caption'] = data['caption'].apply(lambda x: x.replace("\s+", " "))  # Replace multiple spaces
    data['caption'] = data['caption'].apply(lambda x: " ".join([word for word in x.split() if len(word) > 1]))  # Remove single-char words
    data['caption'] = "startseq " + data['caption'] + " endseq"  # Add special tokens to mark sequence start and end
    return data

# Apply preprocessing to captions
data = text_preprocessing(data)
captions = data['caption'].tolist()
captions[:10]

# Tokenize text data for model input
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)  # Fit tokenizer on caption text
vocab_size = len(tokenizer.word_index) + 1  # Vocabulary size, +1 for padding/out-of-vocabulary handling
max_length = max(len(caption.split()) for caption in captions)  # Max caption length for padding purposes

# Split data into training and validation sets
images = data['image'].unique().tolist()
nimages = len(images)
split_index = round(0.85 * nimages)  # 85-15 train-validation split
train_images = images[:split_index]
val_images = images[split_index:]

train = data[data['image'].isin(train_images)].reset_index(drop=True)
test = data[data['image'].isin(val_images)].reset_index(drop=True)

print("Extracting features from images...")

# Feature extraction from images using pre-trained CNN (DenseNet201 in this case)
# TOCHANGE
model = VGG16()
# model = DenseNet201()
fe = Model(inputs=model.input, outputs=model.layers[-2].output)  # Extract features from second-last layer

# Dictionary to store image features
img_size = 224
features = {}
for image in tqdm(data['image'].unique().tolist()):
    img = load_img(os.path.join(image_path, image), target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0  # Normalize image
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    feature = fe.predict(img, verbose=0)  # Extract features
    features[image] = feature

# Define CustomDataGenerator class for generating batches of data
class CustomDataGenerator(Sequence):
    
    def __init__(self, df, X_col, y_col, batch_size, directory, tokenizer, 
                 vocab_size, max_length, features, shuffle=True):
        self.df = df.copy()
        self.X_col = X_col
        self.y_col = y_col
        self.directory = directory
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.features = features
        self.shuffle = shuffle
        self.n = len(self.df)
        
    def on_epoch_end(self):
        if self.shuffle:
            self.df = self.df.sample(frac=1).reset_index(drop=True)
    
    def __len__(self):
        return self.n // self.batch_size
    
    def __getitem__(self, index):
        batch = self.df.iloc[index * self.batch_size:(index + 1) * self.batch_size, :]
        X1, X2, y = self.__get_data(batch)        
        return (X1, X2), y
    
    def __get_data(self, batch):
        X1, X2, y = list(), list(), list()
        
        images = batch[self.X_col].tolist()
           
        for image in images:
            feature = self.features[image][0]
            
            captions = batch.loc[batch[self.X_col] == image, self.y_col].tolist()
            for caption in captions:
                seq = self.tokenizer.texts_to_sequences([caption])[0]

                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=self.vocab_size)[0]
                    X1.append(feature)
                    X2.append(in_seq)
                    y.append(out_seq)
            
        X1, X2, y = np.array(X1), np.array(X2), np.array(y)
        return X1, X2, y
    
#################################
#            MODEL              #
#################################
print("Building model...")

# Define the image captioning model architecture
# TOCHANGE
# Dense
# input1 = Input(shape=(1920,))
# VGG16
input1 = Input(shape=(4096,))  # Wektory cech obrazu
img_features = Dense(512)(input1)  # Zwiększenie liczby jednostek w warstwie Dense
img_features = LeakyReLU(alpha=0.2)(img_features)  # Zastosowanie Leaky ReLU
img_features = BatchNormalization()(img_features)  # Normalizacja cech
img_features = Dropout(0.2)(img_features)  # Regularizacja
# Zmienianie kształtu cech obrazu w celu użycia w mechanizmie uwagi
img_features_reshaped = Reshape((1, 512))(img_features)

# Wejście sekwencji tekstowej
input2 = Input(shape=(max_length,))  # Długość sekwencji (np. maksymalna długość opisu)
sentence_features = Embedding(vocab_size, 256)(input2)  # Większa liczba wymiarów w warstwie osadzającej
# Użycie Bidirectional LSTM (zwiększenie liczby jednostek w LSTM)
sentence_features = Bidirectional(LSTM(256, return_sequences=True))(sentence_features)  # Zwiększenie liczby jednostek w LSTM
# Mechanizm uwagi - MultiHeadAttention
attention = MultiHeadAttention(num_heads=8, key_dim=64)(sentence_features, sentence_features)  # Mechanizm uwagi

# Łączenie cech obrazu i uwagi
merged = Concatenate(axis=1)([sentence_features, attention])  # Łączenie cech obrazu z uwagą
sentence_features = LSTM(512)(merged)  # Zwiększenie liczby jednostek w LSTM
sentence_features = Dropout(0.3)(sentence_features)  # Dropout w tej warstwie

# Dodanie połączenia rezydualnego
x = Add()([sentence_features, img_features])  # Połączenie rezydualne z cechami obrazu

# Zwiększenie rozmiaru warstwy Dense
x = Dense(512)(x)
x = LeakyReLU(alpha=0.2)(x)  # Leaky ReLU
x = Dropout(0.2)(x)  # Dropout

# Finalna warstwa wyjściowa z aktywacją softmax
output = Dense(vocab_size, activation='softmax')(x)

# Define the model
caption_model = Model(inputs=[input1, input2], outputs=output)
# Compile the model
caption_model.compile(loss='categorical_crossentropy', optimizer='adam')
caption_model.summary()

#################################
#       Generating sets         #
#################################

# Create training and validation data generators
train_generator = CustomDataGenerator(df=train, X_col='image', y_col='caption', batch_size=64,
                                      directory=image_path, tokenizer=tokenizer,
                                      vocab_size=vocab_size, max_length=max_length, features=features)

validation_generator = CustomDataGenerator(df=test, X_col='image', y_col='caption', batch_size=64,
                                           directory=image_path, tokenizer=tokenizer,
                                           vocab_size=vocab_size, max_length=max_length, features=features)

#################################
#          TRAINING             #
#################################
print("Training model...")

# Define model checkpoints, early stopping, and learning rate reduction
# TOCHANGE
model_name = "VGG16_model.keras"
checkpoint = ModelCheckpoint(model_name, monitor="val_loss", mode="min", save_best_only=True, verbose=1)
earlystopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=1, restore_best_weights=True)
learning_rate_reduction = ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.2, min_lr=1e-8)

# Train the model using the fit method
history = caption_model.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator,
    callbacks=[checkpoint, earlystopping, learning_rate_reduction]
)

# Save tokenizer as JSON
print("Saving tokenizer as json...")
tokenizer_json = tokenizer.to_json()
# TOCHANGE
with open("VGG16_tokenizer.json", "w") as json_file:
    json_file.write(tokenizer_json)

print("Tokenizer saved successfully.")


print("Plotting training history...")

# Plot training and validation loss over epochs
plt.figure(figsize=(20, 8))
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

#################################
#   Prediction and Evaluation   #
#################################

print("Starting prediction and evaluation...")

# Convert integer token back to word
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

# Predict a caption for an image
def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        
        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)
        
        word = idx_to_word(y_pred, tokenizer)
        
        if word is None:
            break
        in_text += " " + word
        
        if word == 'endseq':
            break
            
    return in_text

for i in range(10):
    # Sample predictions for a random batch of images from the test set
    samples = test.sample(15)
    samples.reset_index(drop=True, inplace=True)

    print(f"Generating {i} captions for sample images...")

    for index,record in samples.iterrows():
        img = load_img(os.path.join(image_path,record['image']),target_size=(224,224))
        img = img_to_array(img)
        img = img/255.
        
        caption = predict_caption(caption_model, record['image'], tokenizer, max_length, features)
        samples.loc[index,'caption'] = caption
    
    display_images(samples)
    plt.show()