"""
A basic transformer for language.
"""
import math
import torch 
import os
import torch.nn as nn
from dotenv import load_dotenv
from tokenizer import build_tokenizer
from transformer_config import TransformerConfig
load_dotenv()
test_text_file = os.getenv("TEXT_FILE", "")



if __name__ == "__main__":
    print("---- Basic Language Transformer ----")

    cfg = TransformerConfig()
    print(f"Transformer Configurations:\n{cfg}")
    

    # Building the vocabulary from the text file
    # IRL this can be done from dictionary, a set of files, etc.
    print(f"Tokenizing a text file - encoding and decoding")
    with open(test_text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    encoder, decoder, vocab_size = build_tokenizer(text)
    print(f"Text file: {test_text_file}")
    print(f"Tokens size:\n {vocab_size}")

    ## Split into train and validation set
    data = torch.tensor(encoder(text), dtype=torch.long)
    print(f"Encoded Text Data Size: {data.size()} Device: {data.device}")
    train_val_splits = [.9, .1]
    num_train = math.floor(train_val_splits[0] * len(data))
    training_sample = data[:num_train]
    validation_sample = data[num_train:]

    print(f"Training Sample:\n{training_sample}")
    print(f"Validation Sample:\n{validation_sample}")
