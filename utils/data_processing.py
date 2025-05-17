import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

def load_captions(filepath):
    """Load captions from a file"""
    captions = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        for line in file:
            tokens = line.strip().split('\t')
            if len(tokens) == 2:
                image_id, caption = tokens
                if image_id not in captions:
                    captions[image_id] = []
                captions[image_id].append(caption)
    return captions

def preprocess_captions(captions):
    """Preprocess captions for training"""
    # Add start and end sequence tokens
    processed_captions = []
    for caption_list in captions.values():
        for caption in caption_list:
            processed_captions.append(f'startseq {caption} endseq')
    return processed_captions

def create_tokenizer(captions, num_words=10000):
    """Create a tokenizer fitted on captions"""
    tokenizer = Tokenizer(num_words=num_words, oov_token='<unk>')
    tokenizer.fit_on_texts(captions)
    return tokenizer

def get_max_length(captions):
    """Get the maximum caption length"""
    return max(len(caption.split()) for caption in captions)

def prepare_sequences(tokenizer, captions, max_length):
    """Prepare sequences for training"""
    X1, X2, y = [], [], []
    for caption in captions:
        seq = tokenizer.texts_to_sequences([caption])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
            out_seq = to_categorical([out_seq], num_classes=len(tokenizer.word_index)+1)[0]
            X1.append(in_seq)
            X2.append(out_seq)
    return np.array(X1), np.array(X2)
