from transformers import AutoModelForSequenceClassification, AutoTokenizer

model_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"
tokenizer_path = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Load model and tokenizer
model = AutoModelForSequenceClassification.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

# Save model and tokenizer
model.save_pretrained("C:\\Users\\Admin\\ML-Models")
tokenizer.save_pretrained("C:\\Users\\Admin\\ML-Models")