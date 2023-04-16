import os
import requests
import collections
import torch
import torchtext 
import json 

import utils 

#
# Preprocess the book
# - download the book from Project Gutenberg if not yet done
# - tokenize the text
# - build a vocabulary from the token
# - encode book
# - save vocabulary 
# - split data into one training data set and one validation data set and save both as JSON files

BOOK_ID = 2600
TRAIN_DATA = "train.json"
VAL_DATA = "val.json"
VOCAB_FILE = "vocab.pt"

#
# Download book and return content
#
def get_book(book_id):
    url = f"https://www.gutenberg.org/files/{book_id}/{book_id}-0.txt"
    dest_file = f"{book_id}.txt"
    if not os.path.exists(dest_file):
        response = requests.get(url)
        f = open(dest_file, "wb")
        f.write(response.content)
        f.close()
    f = open(dest_file, "r")
    content = f.read(-1)
    f.close()
    return content

#
# Build a character-level vocabulary from the token
#
def build_vocab(token):
    counter = collections.Counter(token)
    vocab = torchtext.vocab.build_vocab_from_iterator(counter, specials=["<unk>"])
    vocab.set_default_index(0)
    return vocab


if __name__ == '__main__':
    book = get_book(BOOK_ID)
    token = utils.tokenize(book)
    vocab = build_vocab(token)
    #
    # Save vocab
    #
    torch.save(vocab, VOCAB_FILE)
    #
    # Encode book
    #
    encoded_book = [vocab[t] for t in token]
    #
    # Split into train and validation set
    #
    total_length = len(encoded_book)
    train_length = int(0.9*total_length)
    print(f"Splitting file into {train_length} training items and {total_length - train_length} validation items")
    with open(TRAIN_DATA, "w") as f:
        json.dump(encoded_book[:train_length], f) 
    with open(VAL_DATA, "w") as f:
        json.dump(encoded_book[train_length:], f) 

