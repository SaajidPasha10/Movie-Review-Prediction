import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.datasets import imdb
from tensorflow.keras.utils import pad_sequences

word_to_index = imdb.get_word_index()
index_to_word = {v:k for k,v in word_to_index.items()}

def decode_review(encoded_review):
  decoded_review = [index_to_word.get(i-3,"?") for i in encoded_review]
  decoded_review = " ".join(decoded_review)
  return decoded_review

def preprocess_text(text):
  words = text.lower()
  encoded_review = [word_to_index.get(word,2) + 3 for word in words]
  padded_review = pad_sequences([encoded_review],maxlen=500)
  return padded_review

#load the model

model = load_model("RNN_model.h5")

def predict(text):
  preprocessed_text = preprocess_text(text)
  prob = model.predict(preprocessed_text)
  sentiment ="Positive" if  prob[0][0] > 0.5  else "Negative"
  return sentiment, prob[0][0]
# sentiment,prob = predict("The movie was disaster")
# print(f"Sentiment : {sentiment}, Score: {prob*100}")

st.title("IMDB - Movie Review Classification")
st.write("Enter a text to classify the sentiment")
user_input = st.text_input(label="Enter a review")
if st.button("Classify") and len(user_input) > 5:
  sentiment,prob = predict(user_input)
  prob = round(prob,2) * 100
  st.write(f"Sentiment: {sentiment}")
  st.write(f"Score: {prob} %")

