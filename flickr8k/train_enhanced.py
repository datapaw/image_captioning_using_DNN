import os
import numpy as np
import pandas as pd
from tensorflow.keras.utils import Sequence
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Dense, Embedding, Bidirectional, LSTM, Dropout, BatchNormalization,
    MultiHeadAttention, Concatenate, Add, Reshape
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import warnings

warnings.filterwarnings('ignore')

#################################
#       Utility Functions       #
#################################

def load_dataset(image_dir, captions_file):
    """Load dataset of image paths and captions."""
    data = pd.read_csv(captions_file)
    data['image'] = data['image'].apply(lambda x: os.path.join(image_dir, x))
    return data

def preprocess_captions(data):
    """Preprocess text captions."""
    data['caption'] = (
        data['caption'].str.lower()
        .str.replace(r'[^a-z\s]', '', regex=True)  # Remove special characters
        .str.replace(r'\s+', ' ', regex=True)    # Remove extra spaces
        .str.strip()
    )
    data['caption'] = "startseq " + data['caption'] + " endseq"
    return data

def tokenize_captions(data, max_vocab_size=None):
    """Tokenize text captions."""
    tokenizer = Tokenizer(num_words=max_vocab_size)
    tokenizer.fit_on_texts(data['caption'])
    vocab_size = len(tokenizer.word_index) + 1
    max_length = max(data['caption'].apply(lambda x: len(x.split())))
    return tokenizer, vocab_size, max_length

def extract_features(image_paths, model, target_size=(224, 224)):
    """Extract image features using pre-trained model."""
    features = {}
    for img_path in tqdm(image_paths, desc="Extracting Features"):
        img = load_img(img_path, target_size=target_size)
        img = img_to_array(img) / 255.0
        img = np.expand_dims(img, axis=0)
        feature = model.predict(img, verbose=0)
        features[img_path] = feature
    return features

#################################
#         Model Creation        #
#################################

def build_model(vocab_size, max_length, feature_dim=512):
    """Build the image captioning model."""
    # Image feature input
    img_input = Input(shape=(feature_dim,), name="Image_Input")
    img_features = Dense(256, activation="relu")(img_input)
    img_features = BatchNormalization()(img_features)
    img_features = Dropout(0.2)(img_features)

    # Text input
    text_input = Input(shape=(max_length,), name="Text_Input")
    text_features = Embedding(vocab_size, 256)(text_input)
    text_features = Bidirectional(LSTM(256, return_sequences=True))(text_features)
    attention_output = MultiHeadAttention(num_heads=4, key_dim=64)(text_features, text_features)

    # Merge image and text features
    combined_features = Concatenate()([img_features, attention_output])
    lstm_output = LSTM(256)(combined_features)

    # Dense layers
    dense_output = Dense(256, activation="relu")(lstm_output)
    dense_output = Dropout(0.3)(dense_output)
    final_output = Dense(vocab_size, activation="softmax")(dense_output)

    # Define model
    model = Model(inputs=[img_input, text_input], outputs=final_output)
    model.compile(loss="categorical_crossentropy", optimizer=Adam(learning_rate=0.001))
    return model

#################################
#         Training Setup        #
#################################

def create_generators(data, tokenizer, max_length, features, batch_size=32):
    """Create data generators for training and validation."""
    class CustomDataGenerator(Sequence):
        def __init__(self, data, features, tokenizer, batch_size, max_length, vocab_size):
            self.data = data
            self.features = features
            self.tokenizer = tokenizer
            self.batch_size = batch_size
            self.max_length = max_length
            self.vocab_size = vocab_size

        def __len__(self):
            return len(self.data) // self.batch_size

        def __getitem__(self, idx):
            batch = self.data.iloc[idx * self.batch_size:(idx + 1) * self.batch_size]
            X_img, X_seq, y = [], [], []
            for _, row in batch.iterrows():
                feature = self.features[row['image']][0]
                caption = self.tokenizer.texts_to_sequences([row['caption']])[0]
                for i in range(1, len(caption)):
                    in_seq, out_seq = caption[:i], caption[i]
                    in_seq = pad_sequences([in_seq], maxlen=self.max_length)[0]
                    out_seq = np.zeros(self.vocab_size)
                    out_seq[out_seq] = 1
                    X_img.append(feature)
                    X_seq.append(in_seq)
                    y.append(out_seq)
            return [np.array(X_img), np.array(X_seq)], np.array(y)

    generator = CustomDataGenerator(data, features, tokenizer, batch_size, max_length, vocab_size)
    return generator

#################################
#           Training            #
#################################

# Configuration
IMAGE_DIR = "./dataset/Images"
CAPTIONS_FILE = "./dataset/captions.txt"
BATCH_SIZE = 64
EPOCHS = 50

# Load and preprocess data
data = load_dataset(IMAGE_DIR, CAPTIONS_FILE)
data = preprocess_captions(data)
tokenizer, vocab_size, max_length = tokenize_captions(data)

# Split dataset
image_files = data['image'].unique()
split_idx = int(len(image_files) * 0.8)
train_files = image_files[:split_idx]
val_files = image_files[split_idx:]

train_data = data[data['image'].isin(train_files)]
val_data = data[data['image'].isin(val_files)]

# Feature extraction
pretrained_model = VGG16(include_top=False, pooling="avg")
features = extract_features(image_files, pretrained_model)

# Generators
train_generator = create_generators(train_data, tokenizer, max_length, features, BATCH_SIZE)
val_generator = create_generators(val_data, tokenizer, max_length, features, BATCH_SIZE)

# Model
model = build_model(vocab_size, max_length)

# Callbacks
callbacks = [
    ModelCheckpoint("best_model.h5", save_best_only=True),
    EarlyStopping(patience=5, restore_best_weights=True),
    ReduceLROnPlateau(factor=0.2, patience=3)
]

# Train
history = model.fit(train_generator, validation_data=val_generator, epochs=EPOCHS, callbacks=callbacks)
