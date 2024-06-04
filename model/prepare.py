import sys
sys.path.append('../')
import torch
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from utils.dataloader import DatasetClassify
from Dataset.AGNewsDataset import AGNewsDataset
from Token import load_tokenizer
from torch.utils.data import Dataset, DataLoader

# Instantiate a tokenizer based on a pre-trained model (e.g., BERT)
tokenizer = load_tokenizer()

def prepare_data():
    dataset = DatasetClassify('../data/')
    print("Downloading and Preprocessing the dataset")
    train_df, test_df = dataset.download()

    # Tokenize the input text
    train_encodings = tokenizer(train_df['text'].tolist(), truncation=True, padding=True)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True)

    # Subtract 1 from the labels to match the model's requirements
    train_labels = train_df['label'].values - 1
    test_labels = test_df['label'].values - 1

    # Randomly sample a subset of the original dataset for training
    train_df_sample = train_df.sample(frac=0.01, random_state=42)

    # Tokenize the text in the sampled train dataframe and the test dataframe
    train_encodings = tokenizer(train_df_sample['text'].tolist(), truncation=True, padding=True, max_length=256)
    test_encodings = tokenizer(test_df['text'].tolist(), truncation=True, padding=True, max_length=256)

    # Convert the labels into numerical format using the sampled train dataframe
    train_labels = train_df_sample['label'].values -1
    test_labels = test_df['label'].values-1

    print("Creating DataLoader objects")
    # Create dataset objects for the sampled train data and test data
    train_dataset = AGNewsDataset(train_encodings, train_labels)
    test_dataset = AGNewsDataset(test_encodings, test_labels)

    # Create DataLoader objects for the sampled train data and test data
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    return train_loader, test_loader



