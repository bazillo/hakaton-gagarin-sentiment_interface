import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import AdamW
from sklearn.metrics import accuracy_score, f1_score
from constants import *
from model import TransformerClassificationModel




class TransformerClassificationTrainer:
    def __init__(self, model, tokenizer, max_length=MAX_LENGTH, batch_size=BATCH_SIZE, lr=LR,
                 num_epochs=NUM_EPOCHS, freeze_backbone=FREEZE_BACKBONE):
        self.model = freeze_backbone_function(model, freeze_backbone)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.batch_size = batch_size
        self.lr = lr
        self.num_epochs = num_epochs

    def preprocess_data(self, texts, aspects, labels):
        #inputs = self.tokenizer(texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt")
        inputs = self.tokenizer([wrap(text, aspect) for text, aspect in zip(texts, aspects)],
                                max_length=self.max_length, truncation=True, padding=True, return_tensors="pt")
        labels = torch.tensor(labels)
        return inputs, labels

    def train(self, train_texts, train_aspects, train_labels):
        train_inputs, train_labels = self.preprocess_data(train_texts, train_aspects, train_labels)
        train_data = torch.utils.data.TensorDataset(train_inputs['input_ids'], train_inputs['attention_mask'], train_labels)
        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)

        optimizer = AdamW(self.model.parameters(), lr=self.lr)
        loss_fn = nn.CrossEntropyLoss()

        self.model.to(device)
        self.model.train()

        for epoch in range(self.num_epochs):
            total_loss = 0
            for batch in train_loader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch

                optimizer.zero_grad()
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                loss = loss_fn(logits, labels)
                total_loss += loss.item()

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()

            avg_train_loss = total_loss / len(train_loader)
            print(f"Epoch {epoch+1}/{self.num_epochs}, Train Loss: {avg_train_loss:.4f}")

        return self.model

    def evaluate(self, test_texts, test_aspects, test_labels):
        self.model.eval()
        test_inputs, test_labels = self.preprocess_data(test_texts, test_aspects, test_labels)
        test_data = torch.utils.data.TensorDataset(test_inputs['input_ids'], test_inputs['attention_mask'], test_labels)
        test_loader = DataLoader(test_data, batch_size=self.batch_size, shuffle=False)

        predictions = []
        true_labels = []

        with torch.no_grad():
            for batch in test_loader:
                batch = tuple(t.to(device) for t in batch)
                input_ids, attention_mask, labels = batch
                outputs = self.model(input_ids, attention_mask)
                logits = outputs['logits']
                preds = torch.argmax(logits, dim=1).cpu().numpy()
                predictions.extend(preds)
                true_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(true_labels, predictions)
        print(f"Accuracy: {accuracy:.4f}")
        f1 = f1_score(true_labels, predictions, average='macro')
        print(f"F1 score: {f1:.4f}")

def freeze_backbone_function(model: TransformerClassificationModel, freeze=True):
    for param in model.backbone.parameters():
        param.requires_grad = not freeze
    return model