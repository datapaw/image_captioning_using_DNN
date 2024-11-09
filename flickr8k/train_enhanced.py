import numpy as np
import pandas as pd
import os
import tensorflow as tf
from tqdm import tqdm
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical, Sequence
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Dense, Dropout, Embedding, LSTM, Input, concatenate,
                                     GlobalAveragePooling2D, BatchNormalization)
from tensorflow.keras.applications import DenseNet201, VGG16
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (ModelCheckpoint, EarlyStopping, ReduceLROnPlateau,
                                        CSVLogger, TensorBoard)
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from textwrap import wrap
from pathlib import Path
import json

# Set display preferences
plt.rcParams['font.size'] = 12
sns.set_style("dark")
warnings.filterwarnings('ignore')

#################################
#      Configuration            #
#################################

# Dataset paths
DATASET_PATH = Path('./dataset')
IMAGE_PATH = DATASET_PATH / 'Images'
CAPTIONS_PATH = DATASET_PATH / 'captions.txt'

# Model hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 64
EPOCHS = 50
EMBEDDING_DIM = 128
LSTM_UNITS = 128
DROPOUT_RATE = 0.5
LEARNING_RATE = 0.0005

# Choose model architecture
# TOCHANGE
FEATURE_EXTRACTOR = "DenseNet201"

# Define paths for saved models and logs
MODEL_PATH = Path("saved_models")
MODEL_PATH.mkdir(exist_ok=True)
LOG_PATH = Path("logs")
LOG_PATH.mkdir(exist_ok=True)

#################################
#       Data Loading            #
#################################

print("Loading and preparing dataset...")

# Load captions
data = pd.read_csv(CAPTIONS_PATH)

# Define image loading function
def read_image(path, img_size=IMG_SIZE):
    img = load_img(path, color_mode='rgb', target_size=(img_size, img_size))
    img = img_to_array(img) / 255.0
    return img

# Display sample images
def display_images(temp_df):
    temp_df = temp_df.reset_index(drop=True)
    plt.figure(figsize=(20, 20))
    for i in range(15):
        plt.subplot(5, 5, i + 1)
        plt.subplots_adjust(hspace=0.7, wspace=0.3)
        image = read_image(f"{IMAGE_PATH}/{temp_df.image[i]}")
        plt.imshow(image)
        plt.title("\n".join(wrap(temp_df.caption[i], 20)))
        plt.axis("off")

display_images(data.sample(15))

#################################
#    Text Preprocessing         #
#################################

# Clean and prepare text data
def text_preprocessing(data):
    data['caption'] = data['caption'].str.lower().replace("[^a-z\s]", " ", regex=True).str.strip()
    data['caption'] = data['caption'].apply(lambda x: "startseq " + " ".join(word for word in x.split() if len(word) > 1) + " endseq")
    return data

# Apply text preprocessing to captions
data = text_preprocessing(data)
captions = data['caption'].tolist()

# Tokenize text data for model input
tokenizer = Tokenizer()
tokenizer.fit_on_texts(captions)
vocab_size = len(tokenizer.word_index) + 1
max_length = max(len(caption.split()) for caption in captions)

# Save tokenizer for later use
with open(MODEL_PATH / 'tokenizer.json', 'w') as f:
    f.write(tokenizer.to_json())

#################################
#   Train-Validation Split      #
#################################

images = data['image'].unique().tolist()
split_index = round(0.85 * len(images))
train_images, val_images = images[:split_index], images[split_index:]

train = data[data['image'].isin(train_images)].reset_index(drop=True)
test = data[data['image'].isin(val_images)].reset_index(drop=True)

#################################
#   Feature Extraction          #
#################################
# TOCHANGE

def build_feature_extractor(model_name='DenseNet201'):
    models = {
        "DenseNet201": DenseNet201,
        "VGG16": VGG16
    }
    base_model = models[model_name](weights='imagenet', include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    for layer in base_model.layers[:-10]:  # Unfreeze last few layers
        layer.trainable = False
    return Model(inputs=base_model.input, outputs=GlobalAveragePooling2D()(base_model.output))

print(f"Extracting features using {FEATURE_EXTRACTOR}...")
fe = build_feature_extractor(FEATURE_EXTRACTOR)
features = {}
for image in tqdm(data['image'].unique().tolist()):
    img = load_img(os.path.join(IMAGE_PATH, image), target_size=(IMG_SIZE, IMG_SIZE))
    img = img_to_array(img) / 255.0
    img = np.expand_dims(img, axis=0)
    features[image] = fe.predict(img, verbose=0)

#################################
#  Data Generator Definition    #
#################################

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
        X1, X2, y = [], [], []
        
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
            
        return np.array(X1), np.array(X2), np.array(y)

#################################
#  Model Architecture Setup     #
#################################

print("Building model...")

# Define the model architecture
input1 = Input(shape=(fe.output_shape[1],))
input2 = Input(shape=(max_length,))

img_features = Dense(128, activation='relu')(input1)
img_features = BatchNormalization()(img_features)
img_features = Dropout(DROPOUT_RATE)(img_features)

sentence_features = Embedding(vocab_size, EMBEDDING_DIM, mask_zero=True)(input2)
sentence_features = LSTM(LSTM_UNITS, dropout=0.3, recurrent_dropout=0.3, return_sequences=True)(sentence_features)
sentence_features = LSTM(LSTM_UNITS, dropout=0.3, recurrent_dropout=0.3)(sentence_features)

x = concatenate([img_features, sentence_features])
x = Dense(128, activation='relu')(x)
x = Dropout(DROPOUT_RATE)(x)
output = Dense(vocab_size, activation='softmax')(x)

caption_model = Model(inputs=[input1, input2], outputs=output)
optimizer = Adam(learning_rate=LEARNING_RATE)
caption_model.compile(loss='categorical_crossentropy', optimizer=optimizer)
caption_model.summary()

#################################
#       Data Generators         #
#################################

train_generator = CustomDataGenerator(train, 'image', 'caption', BATCH_SIZE, IMAGE_PATH, tokenizer, vocab_size, max_length, features)
validation_generator = CustomDataGenerator(test, 'image', 'caption', BATCH_SIZE, IMAGE_PATH, tokenizer, vocab_size, max_length, features)

#################################
#   Training and Callbacks      #
#################################

print("Training model...")
# TOCHANGE
callbacks = [
    ModelCheckpoint(MODEL_PATH / "best_model.keras", monitor="val_loss", mode="min", save_best_only=True, verbose=1),
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', patience=2, factor=0.5, min_lr=1e-7, verbose=1),
    CSVLogger(LOG_PATH / 'training_log.csv', append=True),
    TensorBoard(log_dir=LOG_PATH)
]

history = caption_model.fit(
    train_generator,
    epochs=EPOCHS,
    validation_data=validation_generator,
    callbacks=callbacks
)

#################################
#   Plot Loss Curves            #
#################################

plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

#################################
#   Prediction and Evaluation   #
#################################

# Prediction functions for evaluation
def idx_to_word(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(model, image, tokenizer, max_length, features):
    feature = features[image]
    in_text = "startseq"
    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], max_length)
        
        y_pred = model.predict([feature, sequence])
        y_pred = np.argmax(y_pred)
        word = idx_to_word(y_pred, tokenizer)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break
    return in_text

# Test model predictions with sample images
for i in range(10):
    for image in test['image'].sample(5):
        print("Image:", image)
        plt.imshow(read_image(os.path.join(IMAGE_PATH, image)))
        plt.axis("off")
        plt.show()
        
        actual_caption = test[test['image'] == image]['caption'].values[0]
        print("Actual Caption:", actual_caption)
        print("Predicted Caption:", predict_caption(caption_model, image, tokenizer, max_length, features))
        print("\n")
