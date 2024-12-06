import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load models
models = {}
labels = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

for label in labels:
    with open(f"{label}_model.pkl", "rb") as file:
        models[label] = pickle.load(file)

# Load TF-IDF vectorizer
# Note: Save your vectorizer as well during training and load it here
with open("vectorizer.pkl", "rb") as file:
    vectorizer = pickle.load(file)

# Streamlit App
st.title("Harmful text Classification")
st.write("Enter a text message below to classify it into the following categories: toxic, severe toxic, obscene, threat, insult, identity hate.")

# User Input
user_input = st.text_area("Enter a sample test message:", "")

if st.button("Classify"):
    if user_input.strip():
        # Transform user input using the vectorizer
        input_vectorized = vectorizer.transform([user_input])
        
        # Predict for each label
        results = {}
        for label, model in models.items():
            prediction = model.predict(input_vectorized)[0]
            results[label] = "Yes" if prediction == 1 else "No"
        
        # Display Results
        st.subheader("Classification Results:")
        for label, result in results.items():
            st.write(f"{label.capitalize()}: {result}")
    else:
        st.error("Please enter a comment to classify.")
