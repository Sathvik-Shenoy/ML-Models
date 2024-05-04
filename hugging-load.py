from transformers import pipeline, AutoModelForSequenceClassification, AutoTokenizer

# Load saved model and tokenizer
saved_model_path = "C:\\Users\\Admin\\ML-Models"
saved_tokenizer_path = "C:\\Users\\Admin\\ML-Models"

model = AutoModelForSequenceClassification.from_pretrained(saved_model_path)
tokenizer = AutoTokenizer.from_pretrained(saved_tokenizer_path)

# Now you can use the loaded model and tokenizer as before
sentiment_task = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer)
result = sentiment_task("fk")
print(result)