import streamlit as st
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# load the gru model

model= load_model('gru_next_word_prediction_model.h5', compile= False) # Load the trained GRU model from the specified file path without compiling it, as we will use it for prediction only

# load the tokenizer

with open('tokenizer_GRU.pickle', 'rb') as file:
    tokenizer= pickle.load(file)

# Define a function to predict the next word given an input text

def predict_next_word(model, tokenizer, text_sequence, max_sequence_length):
    token_list= tokenizer.texts_to_sequences([text_sequence])[0]    # Convert the input text sequence to a sequence of integers
    if len(token_list)>= max_sequence_length:
        token_list= token_list[-(max_sequence_length-1):] # If the input sequence is longer than the maximum sequence length, truncate it to fit the model's input shape
    token_list= pad_sequences([token_list], maxlen= max_sequence_length-1, padding='pre') # Pad the sequence with zeros at the beginning
    prediction= model.predict(token_list, verbose=0)# Predict the next word using the trained model

    predicted_word_index= np.argmax(prediction, axis=1) # Get the index of the predicted word with the highest probability
    for word, index in tokenizer.word_index.items():
        if index == predicted_word_index:
            return word
    return None # Return None if no word is found for the predicted index

# Streamlit app

st.title("Next Word Prediction Using GRU")
input_text= st.text_input("Enter the Sequenceof Words:") # Get the input text sequence from the user
if st.button("Predict Next Word"):
    max_sequence_length= model.input_shape[1]+1 # Get the maximum sequence length from the model's input shape
    predicted_word= predict_next_word(model, tokenizer, input_text, max_sequence_length) # Predict the next word using the function defined above
    if predicted_word:
        st.write(f"Predicted Next Word: {predicted_word}") # Display the predicted next word to the user
    else:
        st.write("No prediction could be made for the given input.") # Display a message if no prediction could be made
        