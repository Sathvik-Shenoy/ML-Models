def classify_input(inp, vectorizer, model):
    # Transform the input using the fitted vectorizer
    inp_transformed = vectorizer.transform([inp]).toarray()

    # Predict using the loaded model
    prediction = model.predict(inp_transformed)
    
    print("Prediction:", prediction)

if __name__ == "__main__":
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer

    # Load the trained model from the file
    model = joblib.load('trained_model.joblib')

    # Read the input from the file
    with open('input_file.txt', 'r') as file:
        inp_str = file.read()

    # Load the fitted vectorizer from the file
    vectorizer = joblib.load('fitted_vectorizer.joblib')

    # Call the function to classify the input
    classify_input(inp_str, vectorizer, model)