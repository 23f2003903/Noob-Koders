import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
import pickle

def load_model_and_tokenizer(model_path, tokenizer_path):
    """Load the trained model and tokenizer"""
    model = load_model(model_path)
    with open(tokenizer_path, 'rb') as f:
        tokenizer = pickle.load(f)
    return model, tokenizer

def extract_features(image_path):
    """Extract features from image using ResNet50"""
    # Load ResNet50 model for feature extraction
    model = ResNet50(include_top=False, pooling='avg')
    
    # Load and preprocess image
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)
    
    # Extract features
    features = model.predict(img_array, verbose=0)
    return features

def generate_caption(model, tokenizer, photo, max_length=100):
    """Generate caption for an image using the trained model"""
    # Initialize caption with start sequence
    in_text = 'startseq'
    
    # Iterate until end sequence or max length
    for i in range(max_length):
        # Tokenize and pad sequence
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        # Predict next word
        yhat = model.predict([photo, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        # Map index to word
        word = word_for_id(yhat, tokenizer)
        
        # Check for end of caption
        if word is None or word == 'endseq':
            break
            
        # Append word to caption
        in_text += ' ' + word
    
    # Clean up caption (remove start/end tokens)
    caption = in_text.replace('startseq', '').replace('endseq', '').strip()
    return caption

def word_for_id(integer, tokenizer):
    """Map an integer to a word"""
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None
