import os
import datasets
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm.auto import tqdm
import multiprocessing as mp
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from transformers import VisionEncoderDecoderModel, ViTFeatureExtractor, AutoTokenizer, default_data_collator
import random


# Check if GPU is available and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    print(f'GPU Available: {torch.cuda.get_device_name(0)}')
else:
    print('No GPU available, using CPU.')

# Disable Weights and Biases for Kaggle
os.environ["WANDB_DISABLED"] = "true"

# Configuration Class
class config:
    ENCODER = "google/vit-base-patch16-224"
    DECODER = "gpt2"
    TRAIN_BATCH_SIZE = 8
    VAL_BATCH_SIZE = 8
    LR = 5e-5
    SEED = 42
    MAX_LEN = 128
    WEIGHT_DECAY = 0.01
    IMG_SIZE = (224, 224)
    LABEL_MASK = -100
    EPOCHS = 3

# Override tokenizer method to include special tokens
def build_inputs_with_special_tokens(self, token_ids_0, token_ids_1=None):
    return [self.bos_token_id] + token_ids_0 + [self.eos_token_id]
AutoTokenizer.build_inputs_with_special_tokens = build_inputs_with_special_tokens

# Initialize feature extractor and tokenizer
feature_extractor = ViTFeatureExtractor.from_pretrained(config.ENCODER)
tokenizer = AutoTokenizer.from_pretrained(config.DECODER)
tokenizer.pad_token = tokenizer.unk_token  # Set pad_token to unk_token

# Image transformation pipeline (including augmentations)
transform = transforms.Compose(
    [
        transforms.Resize(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Pre-trained model normalization
    ]
)

# Load dataset
df = pd.read_csv("./dataset/captions.txt")
train_df, val_df = train_test_split(df, test_size=0.2, random_state=config.SEED)
df.head()

# Custom Dataset Class for Image-Caption Pair
class ImgDataset(Dataset):
    def __init__(self, df, root_dir, tokenizer, feature_extractor, transform=None):
        self.df = df
        self.root_dir = Path(root_dir)
        self.tokenizer = tokenizer
        self.feature_extractor = feature_extractor
        self.transform = transform
        self.max_length = 50

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        caption = self.df.caption.iloc[idx]
        image = self.df.image.iloc[idx]
        img_path = self.root_dir / image

        # Error handling for image loading
        try:
            img = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            img = Image.new("RGB", config.IMG_SIZE)

        if self.transform:
            img = self.transform(img)

        pixel_values = self.feature_extractor(img, return_tensors="pt").pixel_values

        captions = self.tokenizer(caption, padding="max_length", max_length=self.max_length).input_ids
        captions = torch.where(torch.tensor(captions) != self.tokenizer.pad_token_id, torch.tensor(captions), torch.tensor(config.LABEL_MASK))
        
        encoding = {"pixel_values": pixel_values.squeeze(), "labels": captions}
        return encoding

# Instantiate the datasets
train_dataset = ImgDataset(train_df, root_dir="./dataset/Images", tokenizer=tokenizer, feature_extractor=feature_extractor, transform=transform)
val_dataset = ImgDataset(val_df, root_dir="./dataset/Images", tokenizer=tokenizer, feature_extractor=feature_extractor, transform=transform)

# Create DataLoader with multiple workers for faster data loading
train_loader = DataLoader(train_dataset, batch_size=config.TRAIN_BATCH_SIZE, shuffle=True, num_workers=mp.cpu_count())
val_loader = DataLoader(val_dataset, batch_size=config.VAL_BATCH_SIZE, shuffle=False, num_workers=mp.cpu_count())

# Initialize Model
model = VisionEncoderDecoderModel.from_encoder_decoder_pretrained(config.ENCODER, config.DECODER)
model.config.pad_token_id = tokenizer.pad_token_id
model.config.eos_token_id = tokenizer.sep_token_id
model.config.decoder_start_token_id = tokenizer.bos_token_id
model.config.vocab_size = model.config.decoder.vocab_size
model.config.max_length = config.MAX_LEN
model.config.early_stopping = True
model.config.no_repeat_ngram_size = 3
model.config.length_penalty = 2.0
model.config.num_beams = 4

# Move model to device (GPU/CPU)
model = model.to(device)

# Define Training Arguments
training_args = Seq2SeqTrainingArguments(
    output_dir='VIT_large_gpt2',
    per_device_train_batch_size=config.TRAIN_BATCH_SIZE,
    per_device_eval_batch_size=config.VAL_BATCH_SIZE,
    predict_with_generate=True,
    evaluation_strategy="epoch",
    logging_steps=100,
    save_steps=200,
    warmup_steps=100,
    learning_rate=config.LR,
    num_train_epochs=config.EPOCHS,
    overwrite_output_dir=True,
    save_total_limit=1,
    report_to="none", 
)

# Initialize Trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,  # Necessary for tokenization during evaluation
    data_collator=default_data_collator,
)

# Train the model
train_output = trainer.train()

# Save the trained model
trainer.save_model('VIT_large_gpt2')

# Plot the training and validation loss
metrics = trainer.state.log_history  # Contains loss metrics during training

train_loss = [log["loss"] for log in metrics if "loss" in log]
eval_loss = [log["eval_loss"] for log in metrics if "eval_loss" in log]

plt.figure(figsize=(8, 6))
plt.plot(train_loss, label="Training Loss", color="blue", marker="o")
if eval_loss:
    plt.plot(eval_loss, label="Validation Loss", color="orange", marker="o")
plt.title("Loss During Training")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()
