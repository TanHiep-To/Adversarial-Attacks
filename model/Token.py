from transformers import AutoTokenizer
def load_tokenizer(model_name = 'bert-base-uncased'):
    # Instantiate a tokenizer based on a pre-trained model (e.g., BERT)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    return tokenizer