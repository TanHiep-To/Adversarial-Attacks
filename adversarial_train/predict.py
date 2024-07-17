import torch

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