import sys
sys.path.append('../')
import torch
from model.Token import load_tokenizer
from transformers import AutoModelForSequenceClassification
from utils.dataloader import DatasetClassify
from torch.optim import AdamW
from SimpleTextDataset import SimpleTextDataset
from Dataset.AGNewsDataset import AGNewsDataset
from torch.utils.data import DataLoader
import nltk
import textattack
from textattack.attack_recipes import TextFoolerJin2019
from textattack.models.wrappers import HuggingFaceModelWrapper
import pickle
from textattack.attack_results import SuccessfulAttackResult
import os


print('Downloading NLTK resources...\n\n')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('universal_tagset')


tokenizer = load_tokenizer()
# Select a sub of the test set for generating adversarial examples
print('Downloading and preprocessing the dataset...\n\n')
_,test_df = DatasetClassify('./data/').download()
sub_test_df = test_df.sample(n=100, random_state=42)
sub_test_encodings = tokenizer(sub_test_df['text'].tolist(), truncation=True, padding=True, max_length=256)
sub_test_labels = sub_test_df['label'].values - 1
sub_test_dataset = AGNewsDataset(sub_test_encodings, sub_test_labels)
sub_test_loader = DataLoader(sub_test_dataset, batch_size=16, shuffle=False)

# Instance the SimpleTextDataset class
sub_test_dataset_custom = SimpleTextDataset(sub_test_dataset, tokenizer)

# Wrap the model for TextAttack
model =  AutoModelForSequenceClassification.from_pretrained('../run/weights/')
model.eval()
model_wrapper = HuggingFaceModelWrapper(model, tokenizer)

# Instantiate the TextFooler attack recipe
attack = TextFoolerJin2019.build(model_wrapper)

# Generate adversarial examples
print('Generating adversarial examples...\n\n')

adversarial_examples = []
for i in range(len(sub_test_dataset_custom)):
    print('Generate adversarial example', i)
    original_text, ground_truth_label = sub_test_dataset_custom[i]
    attack_result = attack.attack(original_text, ground_truth_label)
    adversarial_examples.append(attack_result)

# Save the adversarial examples
if not os.path.exists('../output'):
    os.makedirs('../output')

with open('../output/adversarial_examples.pkl', 'wb') as f:
    pickle.dump(adversarial_examples, f)

# Load the  adversarial examples
with open('./output/adversarial_examples.pkl', 'rb') as f:
    adversarial_examples = pickle.load(f)

def compute_accuracy(model, tokenizer, dataset):
    total_count = len(dataset)
    correct_count = 0

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    for text, true_label in dataset:
        inputs = tokenizer(text, return_tensors="pt", padding=True, truncation=True)
        inputs = {key: tensor.to(device) for key, tensor in inputs.items()}  # Move inputs to the same device as the model
        outputs = model(**inputs)
        predicted_label = torch.argmax(outputs.logits, dim=1).item()

        if predicted_label == true_label:
            correct_count += 1

    return correct_count / total_count

# Evaluate the model's performance on the original test set
original_accuracy = compute_accuracy(model, tokenizer, sub_test_dataset_custom)
print("Original Accuracy: {:.2f}%".format(original_accuracy * 100))

# Evaluate the model's performance on the adversarial examples
successful_attacks = [example for example in adversarial_examples if isinstance(example, SuccessfulAttackResult)]
adversarial_dataset = [(example.perturbed_text(), example.original_result.ground_truth_output) for example in successful_attacks]
adversarial_accuracy = compute_accuracy(model, tokenizer, adversarial_dataset)
print("Adversarial Accuracy: {:.2f}%".format(adversarial_accuracy * 100))


# Visualize the adversarial examples
num_examples = 5  # Number of examples to display

print("Case Study: Visualizing the difference between original and adversarial text\n")

for i, attack_result in enumerate(successful_attacks[:num_examples]):
    print(f"Example {i+1}:")
    print("-" * 80)
    print("Original Text:")
    print(attack_result.original_text())
    print("\nAdversarial Text:")
    print(attack_result.perturbed_text())
    print("\nGround Truth Label:", attack_result.original_result.ground_truth_output)
    print("Predicted Label (Original):", attack_result.original_result.output)
    print("Predicted Label (Adversarial):", attack_result.perturbed_result.output)
    print("-" * 80)
    print("\n")

with open('../output/adversarial_examples.txt', 'w') as f:
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