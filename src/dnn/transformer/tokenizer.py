"""
A basic vocabulary builder and tokenizer for languages.
Reads in a file  and a vocabulary of unique characters,
and encodes the text into integers, and decodes integers back to text.
"""

import os
from dotenv import load_dotenv
load_dotenv()

def build_tokenizer(text):
    """
    Builds a tokenizer from the given text,
    returns the encoder, decoder funcs along
    with the vocab size.
    """
    chars = sorted(list(set(text)))
    # Char to integer mapping
    stoi = {ch: i for i, ch in enumerate(chars)}
    # Integer to char mapping
    itos = {i: ch for i, ch in enumerate(chars)}
    vocab_size = len(chars)
    encode = lambda txt: [stoi[ch] for ch in txt]
    decode = lambda txtval: "".join(itos[i] for i in txtval)
    return encode, decode, vocab_size
  
if __name__ == "__main__":
    sample_text = "This is a very basic transformer tokenizer"
    tokenizer = build_tokenizer(sample_text)
    print(f"Tokenizer:\n {tokenizer}")