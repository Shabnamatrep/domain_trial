# multilabeldomain.py
#model training and saving

import pandas as pd
import os
import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertModel
from torch.utils.data import DataLoader, Dataset
from torch.nn.functional import sigmoid

class DistilBertBinaryClassifier(nn.Module):
    def __init__(self):
        super(DistilBertBinaryClassifier, self).__init__()
        self.bert = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.classifier = nn.Linear(self.bert.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        cls_output = outputs[0][:, 0, :]
        logits = self.classifier(cls_output)
        return logits

class CustomDataset(Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['labels'] = self.labels[idx]
        return item

    def __len__(self):
        return len(self.labels)

def train_domain_model(domain_label, df, epochs=5):
    """
    Trains the model for a specific domain.
    
    Parameters:
        domain_label (str): The domain label (e.g., 'w', 'f', 'ps', 'r')
        df (pd.DataFrame): DataFrame containing sentences and labels for the domain.
        epochs (int): Number of training epochs.
    
    Returns:
        The trained model and tokenizer.
    """
    X = df['sentence']
    y = df[f'{domain_label}label']

    y_tensor = torch.tensor(y.values, dtype=torch.float)
    X_train, X_test, y_train, y_test = train_test_split(X, y_tensor, test_size=0.2, random_state=42)

    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    X_train_encodings = tokenizer(list(X_train), truncation=True, padding=True, return_tensors="pt", max_length=128)
    X_test_encodings = tokenizer(list(X_test), truncation=True, padding=True, return_tensors="pt", max_length=128)

    train_dataset = CustomDataset(X_train_encodings, y_train)
    test_dataset = CustomDataset(X_test_encodings, y_test)

    classifier_model = DistilBertBinaryClassifier()
    optimizer = Adam(classifier_model.parameters(), lr=1e-5, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss()

    classifier_model.train()
    for epoch in range(epochs):
        for batch in DataLoader(train_dataset, batch_size=16, shuffle=True):
            optimizer.zero_grad()
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels'].unsqueeze(1)

            outputs = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
            loss = loss_fn(outputs, labels)

            loss.backward()
            optimizer.step()

    classifier_model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in DataLoader(test_dataset, batch_size=16):
            input_ids = batch['input_ids']
            attention_mask = batch['attention_mask']
            labels = batch['labels']

            logits = classifier_model(input_ids=input_ids, attention_mask=attention_mask)
            probs = sigmoid(logits)
            all_probs.extend(probs.cpu().numpy())

            preds = (probs > 0.2).float()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return classifier_model, tokenizer

def save_model_and_tokenizer(model, tokenizer, domain_label):
    model_save_path = f'multilabel_domain_model/distilbert_binary_classifier_{domain_label}.pth'
    tokenizer_save_path = f'multilabel_domain_model/distilbert_binary_tokenizer_{domain_label}'
    tokenizer.save_pretrained(tokenizer_save_path)
    torch.save(model.state_dict(), model_save_path)

def run_all():
    """
    Runs the training and saving of models for all domains ('w', 'f', 'ps', 'r').
    """
    # Read the dataset
    df = pd.read_excel('domain_all.xlsx')
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    df = df.drop(columns=["negative_wellbeing", "negative_functioning"])

    # Define domains to train for
    domains = ['w', 'f', 'ps', 'r']

    # Train and save models for each domain
    for domain in domains:
        print(f"Training model for domain: {domain}")
        model, tokenizer = train_domain_model(domain, df)
        save_model_and_tokenizer(model, tokenizer, domain)
        print(f"Model for domain {domain} saved.")

