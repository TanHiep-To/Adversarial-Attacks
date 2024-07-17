import sys
sys.path.append('./')
import torch
from model.Token import load_tokenizer
from transformers import AutoModelForSequenceClassification
from utils.dataloader import DatasetClassify
from torch.optim import AdamW
from textfoolter.SimpleTextDataset import SimpleTextDataset
from Dataset.AGNewsDataset import AGNewsDataset
from torch.utils.data import DataLoader
import nltk
from tqdm.auto import tqdm
import textattack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
import pickle
from textattack.attack_results import SuccessfulAttackResult
import os
from predict import compute_accuracy

print('Downloading NLTK resources...\n\n')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


tokenizer = load_tokenizer()
dataset = DatasetClassify('./data/')
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


sub_test_df = test_df.sample(n=100, random_state=42)
sub_test_encodings = tokenizer(sub_test_df['text'].tolist(), truncation=True, padding=True, max_length=256)
sub_test_labels = sub_test_df['label'].values - 1
sub_test_dataset = AGNewsDataset(sub_test_encodings, sub_test_labels)
sub_test_loader = DataLoader(sub_test_dataset, batch_size=16, shuffle=False)

# Instance the SimpleTextDataset class
sub_test_dataset_custom = SimpleTextDataset(sub_test_dataset, tokenizer)


train_subset = torch.utils.data.Subset(train_dataset, range(100))  # Use a subset of 100 samples
train_subset_custom = SimpleTextDataset(train_subset, tokenizer)

#Loading model
print('Loading pretrained model .... \n\n')
model =  AutoModelForSequenceClassification.from_pretrained('../run/weights/')
model.eval()
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)


# Instantiate the TextFooler attack recipe
attack = TextFoolerJin2019.build(model_wrapper)

adversarial_train_examples = []

for original_text, ground_truth_label in tqdm(train_subset_custom, desc="Generating adversarial examples"):
    attack_result = attack.attack(original_text, ground_truth_label)
    if isinstance(attack_result, SuccessfulAttackResult):
        adversarial_train_examples.append((attack_result.perturbed_text(), ground_truth_label))

print(adversarial_train_examples)
# Mix the original training dataset with the generated adversarial examples.
mixed_train_dataset = [(text, label) for text, label in train_subset_custom] + adversarial_train_examples




# Move the model to the GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Use a DataLoader to handle batching of the mixed dataset
mixed_train_dataloader = DataLoader(mixed_train_dataset, batch_size=16, shuffle=True)

# Set up the optimizer
optimizer = AdamW(model.parameters(), lr=5e-5)

# Train the model
num_epochs = 100
model.train()

for epoch in range(num_epochs):
    print(f"Epoch {epoch+1}/{num_epochs}")
    for batch in tqdm(mixed_train_dataloader, desc="Training"):
        optimizer.zero_grad()
        texts, labels = batch
        inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}
        labels = torch.tensor(labels).to(device)
        
        outputs = model(**inputs, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()

# Save fine tuning model
if not os.path.exists('./run/weights_optimizer/'):
        os.makedirs('./run/weights_optimizer/')
        
model.save_pretrained('./run/weights_optimizer/')

# Evaluate the model.

# Load the  adversarial examples
with open('./output/adversarial_examples.pkl', 'rb') as f:
    adversarial_examples = pickle.load(f)

# Evaluate the model's performance on the original test set
original_accuracy = compute_accuracy(model, tokenizer, sub_test_dataset_custom)
print("Original Accuracy: {:.2f}%".format(original_accuracy * 100))

# Evaluate the model's performance on the adversarial examples
successful_attacks = [example for example in adversarial_examples if isinstance(example, SuccessfulAttackResult)]
adversarial_dataset = [(example.perturbed_text(), example.original_result.ground_truth_output) for example in successful_attacks]
adversarial_accuracy = compute_accuracy(model, tokenizer, adversarial_dataset)
print("Adversarial Accuracy: {:.2f}%".format(adversarial_accuracy * 100))

with open('./output/adversarial_examples_optimizer.txt', 'w') as f:
    for i, attack_result in enumerate(successful_attacks):  # assuming attack_results is a list of results
        f.write(f"Example {i+1}:\n")
        f.write("-" * 80 + "\n")
        f.write("Original Text:\n")
        f.write(attack_result.original_text() + "\n")
        f.write("\nAdversarial Text:\n")
        f.write(attack_result.perturbed_text() + "\n")
        f.write("\nGround Truth Label: " + str(attack_result.original_result.ground_truth_output) + "\n")
        f.write("Predicted Label (Original): " + str(attack_result.original_result.output) + "\n")
        f.write("Predicted Label (Adversarial): " + str(attack_result.perturbed_result.output) + "\n")
        f.write("-" * 80 + "\n\n")
        
