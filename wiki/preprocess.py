#
# Preprocessing pipeline forWikiText103 dataset
#
# The following steps are performed:
# - if no infiles are specified, download the WikiText103 dataset and concatenate all paragraphs into one large string
# - apply a pretokenizer to the concatenated input
# - train a byte-pair-encoder and save the rules (rules.dat) and vocabulary (vocab.dat) or re-initialize from existing files
# - encode the data and save it as a JSON file
# - finally split the full data into a training set train.json and a validation set val.json
#
# The result will be a list of indices referring to the vocabulary, along with the saved vocabulary and the saved rules
#

import os
import argparse
from tqdm import tqdm
import json
import re
from torchtext.datasets import WikiText103


import BPE
import util

MERGES = 6000
VFILE= "vocab.dat"
RFILE = "rules.dat"
FULL_DATA = "data.json"
TRAIN_DATA = "train.json"
VAL_DATA = "val.json"

#
# Parse arguments
# 
parser = argparse.ArgumentParser();
parser.add_argument("--infile", type=str, default=None, help="Specify input file (if none run unit tests)")
parser.add_argument("--limit", type=int, default=None, help="Number of token to include in output)")
args = parser.parse_args();


#
# STEP 1: download WikiText103 dataset 
#
text = ""
if args.infile is None:
    print("Downloading WikiText103 data set")
    ds = WikiText103(split="train")
    items = [_p for _p in ds]
    print("Cleaning data")
    #
    # Each item is a paragraph as a long string. Clean and concatenate
    # 
    for item in items:
        # Remove trailing whitespace and @
        item = re.sub("^\s+", "", item)
        item = re.sub("@", "", item)
        item = re.sub("\n", "", item)
        if not re.match("^=", item):
            text = text + item

else:
    print(f"Reading text from {args.infile}")
    with open(args.infile, "r") as f:
        text = f.read(-1)

print(f"Got {len(text)} characters in total")


#
# STEP 2: pretokenize to create a list of words
#
print("Pre-tokenizing text")
tokenizer = util.get_tokenizer()
pre_tokenized_text = tokenizer(text)
del text
print(f"Got {len(pre_tokenized_text)} token")

#
# STEP 3: train a BPE tokenizer on the list of words unless we already have a vocabulary and rules
#
encoder = BPE.BPEEncoder(progress_bars = True)
if not (os.path.exists(VFILE) and os.path.exists(RFILE)):
    print("Initializing vocabulary")
    encoder.init_vocab(pre_tokenized_text)
    print(f"Start BPE learning phase ({MERGES} merges)")
    encoder.learn(s = MERGES, align_vocab = True)
    encoder.save_rules(RFILE)
    encoder.save_vocab(VFILE)
else:
    print("Loading vocabulary and rules")
    encoder.load_rules(RFILE)
    encoder.load_vocab(VFILE)
print(f"Vocabulary size is now {len(encoder.get_vocab())}")

#
# STEP 4: encode the input and save the results (or load from file)
#
if not os.path.exists(FULL_DATA):
    if args.limit is not None:
        pre_tokenized_text = pre_tokenized_text[:args.limit]
        print(f"Using only first {args.limit} words for encoding")
    encoded_text = []
    encoder.compile_rules()
    for word in tqdm(pre_tokenized_text, desc="Encoding"):
        encoded_text.extend(encoder.encode(word))
    with open(FULL_DATA, "w") as f:
        json.dump(encoded_text, f) 
else:
    with open(FULL_DATA, "r") as f:
        encoded_text = json.load(f) 

del pre_tokenized_text

#
# STEP 5: optional -run some tests (decode the first 100 symbols and print them)
#
print("Printing first symbols from text")

decoded_text = encoder.decode(encoded_text[:100])
decoded_text = re.sub(r"</w>", " ", decoded_text)
print(decoded_text)

#
# STEP 6: split into a training and a validation set
#
total_length = len(encoded_text)
train_length = int(0.9*total_length)
print(f"Splitting file into {train_length} training items and {total_length - train_length} validation items")
with open(TRAIN_DATA, "w") as f:
    json.dump(encoded_text[:train_length], f) 
with open(VAL_DATA, "w") as f:
    json.dump(encoded_text[train_length:], f) 

