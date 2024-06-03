import sys
sys.path.append('../')
import torch
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from utils.dataloader import DatasetClassify
from Dataset.AGNewsDataset import AGNewsDataset
from prepare import prepare_data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from Token import load_tokenizer
# Choose a pre-trained model architecture (e.g., BERT)
model_name = "bert-base-uncased"

# Instantiate a tokenizer based on a pre-trained model (e.g., BERT)
tokenizer = load_tokenizer()

train_loader, test_loader = prepare_data()

# Define the AGNewsClassifier class
class AGNewsClassifier:
    def __init__(self, model_name, num_labels, learning_rate=2e-5):
        self.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
        self.model.to(self.device)
        self.optimizer = AdamW(self.model.parameters(), lr=learning_rate)
        self.criterion = torch.nn.CrossEntropyLoss()

    def train_epoch(self, dataloader):
        self.model.train()
        running_loss = 0.0
        for batch in dataloader:
            self.optimizer.zero_grad()
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['labels'].to(self.device)
            outputs = self.model(input_ids, attention_mask=attention_mask)
            loss = self.criterion(outputs.logits, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += loss.item()
        return running_loss / len(dataloader)

    def evaluate(self, dataloader):
        self.model.eval()
        total_loss = 0
        correct = 0
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                loss = self.criterion(outputs.logits, labels)
                total_loss += loss.item()
                predictions = torch.argmax(outputs.logits, dim=1)
                correct += (predictions == labels).sum().item()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / len(dataloader.dataset)
        return avg_loss, accuracy

    def predict(self, dataloader):
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                outputs = self.model(input_ids, attention_mask=attention_mask)
                preds = torch.argmax(outputs.logits, dim=1)
                predictions.extend(preds.cpu().numpy())
        return predictions

    def save_model(self, path):
        self.model.save_pretrained(path)

    def load_model(self, path):
        self.model = AutoModelForSequenceClassification.from_pretrained(path)
        self.model.to(self.device)

def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0

    for batch in dataloader:
        optimizer.zero_grad()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.logits, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()

    return running_loss / len(dataloader)

# Evaluate the model's performance on the test set
def evaluate(model, dataloader, device):
    model.eval()
    true_labels = []
    pred_labels = []

    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            _, preds = torch.max(outputs.logits, dim=1)

            true_labels.extend(labels.cpu().numpy())
            pred_labels.extend(preds.cpu().numpy())

    return true_labels, pred_labels



def main():
    # Instantiate the model for sequence classification
    classifier = AGNewsClassifier(model_name, num_labels=4)
    
    # Instantiate the model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=4)
    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = AdamW(model.parameters(), lr=2e-5)

    torch.cuda.empty_cache()
    # Print the model architecture
    # print("\nModel Architecture:")
    # print(model)
    
    
    num_epochs = 1
    print('Training classifier on dataset with', num_epochs, ' epochs')
    # Train the model on the preprocessed dataset for several epochs
    for epoch in range(num_epochs):
        train_loss = classifier.train_epoch(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")
    
    
    # Evaluate the model on the test set
    true_labels, pred_labels = evaluate(classifier.model, test_loader, classifier.device)
    
    # Train the model on the preprocessed dataset for several epochs

    for epoch in range(num_epochs):
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device)
        print(f"Epoch {epoch + 1}/{num_epochs}, Loss: {train_loss:.4f}")
        
    
    # Evaluate the model on the test set
    # Calculate performance metrics
    accuracy = accuracy_score(true_labels, pred_labels)
    precision = precision_score(true_labels, pred_labels, average='weighted')
    recall = recall_score(true_labels, pred_labels, average='weighted')
    f1 = f1_score(true_labels, pred_labels, average='weighted')

    # Print the performance metrics
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")
        
    # Save the trained model
    if not os.path.exists('../model/weights/'):
        os.makedirs('../model/weights/')
        
    model.save_pretrained('../model/weights/')

if __name__ == '__main__':
    main()