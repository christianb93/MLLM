#
# Test our BEP implementation against a straighforward, simple code, 
# which is essentially taken from the original paper
#
# This script can operate in two modes. If no arguments are given, it simply runs a few unit tests, 
# including unit tests against a reference file created using the BPE reference implementation. If an 
# input file is provided the second processing mode is activated in which the encoder simply reads a file
# and creates a set of rules
#
# 
import os
import tempfile
import collections
import re
import argparse
from tqdm import tqdm

import BPE


def get_word_frequencies(pre_tokenized_text):
    counter = collections.Counter(pre_tokenized_text)
    word_frequencies = {" ".join(word) + "</w>" : frequency for word, frequency in counter.items() if len(word) > 0}
    return word_frequencies

def build_vocab(word_frequencies):
    vocab = set()
    for word in word_frequencies.keys():
        for c in word.split():
            vocab.add(c)
    return vocab


def get_stats(word_frequencies):
  pairs = collections.defaultdict(int)
  for word, freq in word_frequencies.items():
    symbols = word.split()
    for i in range(len(symbols)-1):
      pairs[symbols[i],symbols[i+1]] += freq
  return pairs

def read_rules(filename):
    with open(filename, mode = "r") as f:
        #
        # Convert each line into a rule, skipping lines starting with #
        #
        rules = [tuple(rule.split()) for rule in f if not re.match("^#", rule)]
    return rules

def read_and_pretokenize(filename, progress_bar = False):
    text = []
    with open(filename, mode = "r") as f:
        for line in tqdm(f, desc = "Processing file", disable = not progress_bar):
            line = line.strip('\r\n ').split(' ')
            text.extend([w for w in line if len(w) > 0])

    return text


#
# Build vocab for a short text
#
def test1():
   encoder = BPE.BPEEncoder()   
   encoder.init_vocab(["low", "lower",  "newest", "widest"])
   #
   # Vocab should be 
   # l, o, w</w>, w, e, r</w>, n, s, t</w>, w, i, d, </unk>
   #
   _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'r</w>', 'n', 's', 't</w>', 'i', 'd', '</unk>'}
   assert(_vocab == set(encoder.get_vocab()))
   assert(encoder['</unk>'] == encoder.get_default_index())

#
# Build statistics for a short text
#
def test2():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    word_frequencies = get_word_frequencies(pre_tokenized_text)
    vocab = build_vocab(word_frequencies)
    _stats = get_stats(word_frequencies)
    stats = encoder.get_stats()
    assert(stats == _stats)
    #
    # do some checks
    #
    assert(stats['l', 'o'] == 2)
    assert(stats['o', 'w'] == 1)
    assert(stats['o', 'w</w>'] == 1)
    assert(stats['e', 'w'] == 1)


#
# Do one merge
#
def test3():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn()
    #
    # check new vocab - we should have learned the rule w e
    #
    _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'we', 'r</w>', 'n', 's', 't</w>', 'i', 'd', '</unk>'}
    assert(_vocab == set(encoder.get_vocab()))
    #
    # Should have learned one rule
    #
    assert(encoder.get_rules() == [('w', 'e')])


#
# Do two merges
#
def test4():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 2)
    #
    # check new vocab - we should have learned the rule w e and the rule s t</w>
    #
    _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'we', 'r</w>', 'n', 's', 't</w>', 'i', 'd', 'st</w>', '</unk>'}
    assert(_vocab == set(encoder.get_vocab()))
    #
    # Should have learned one rule
    #
    assert(encoder.get_rules() == [('w', 'e'), ('s', 't</w>')])

#
# Do three merges
#
def test5():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 3)
    #
    # check new vocab - we should have learned the rules w e ,  s t</w> and l o
    #
    _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'we', 'r</w>', 'n', 's', 't</w>', 'i', 'd', 'st</w>', 'lo', '</unk>'}
    assert(_vocab == set(encoder.get_vocab()))
    #
    # Should have learned one rule
    #
    assert(encoder.get_rules() == [('w', 'e'), ('s', 't</w>'), ('l', 'o')])

#
# Do a longer test
#
def test6():
    pre_tokenized_text = "this is a longer test than before".split()
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 10)
    assert(encoder.get_rules() == [
        ('t',  'h'),
        ('i', 's</w>'),
        ('th',  'is</w>'),
        ('th',  'a'),
        ('tha',  'n</w>'), 
        ('t', 'e'),
        ('te', 's'),
        ('tes', 't</w>'),
        ('r', 'e</w>'),
        ('o', 're</w>')
    ])

#
# Do a test with a different minimum frequency
#
def test7():
    pre_tokenized_text = "this is a longer test than before".split()
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 10, min_frequency=2)
    assert(encoder.get_rules() == [
        ('t',  'h'),
        ('i', 's</w>')
    ])

#
# An even longer text
#
# This also verifies that when doing the merge "t h " --> "th", we do not accidently merge the t and the h in "w i t h</w>"
#
#
def test8():
    pre_tokenized_text = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation".strip('\r\n ').split(' ')
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 8)
    assert(encoder.get_rules() == [
        ('s', 'i'),
        ('p', 'h'),
        ('t', 'h'),
        ('si', 'g'),
        ('s', 'e</w>'),
        ('r', 'a'),
        ('o', 'n</w>'),
        ('i', 't'),
    ])

#
# Read test data from an input file and compare with a previously generated rules file
# This was generated using the reference implementation with -s=50 --min-frequency=1
#
def test9():
    rules = read_rules("test.rules")
    assert 50 == len(rules), "Expected 50 rules, are you sure you did do the reference run with -s=50?"
    text = read_and_pretokenize("test.in")
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(text)
    encoder.learn(s = 50)
    _rules = encoder.get_rules()
    assert len(_rules) == 50, "Expected 50 rules"
    for i in range(50):
        assert rules[i] == _rules[i], f"Rule {i} does not match - want {rules[i]}, got {_rules[i]}"

#
# Test __getitem__
#
def test10():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 3)
    #
    # check new vocab - we should have learned the rules w e ,  s t</w> and l o
    #
    _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'we', 'r</w>', 'n', 's', 't</w>', 'i', 'd', 'st</w>', 'lo', '</unk>'}
    assert(_vocab == set(encoder.get_vocab()))
    #
    # Get items from vocab
    #
    assert(len(encoder) == len(encoder.get_vocab()))
    vocab = set()
    for index in range(len(encoder)):
        vocab.add(encoder.lookup_token(index))
    assert(vocab == _vocab)

#
# Test encode 
#
def test11():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 3)
    #
    # check new vocab - we should have learned the rules w e ,  s t</w> and l o
    #
    _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'we', 'r</w>', 'n', 's', 't</w>', 'i', 'd', 'st</w>', 'lo', '</unk>'}
    assert(_vocab == set(encoder.get_vocab()))
    #
    # Encode "st" --> should give st
    #
    indices = encoder.encode("st")
    assert(1 == len(indices))
    assert(encoder.lookup_token(indices[0]) == "st</w>")
    #
    # Encode low --> should give ["lo", "w</w>"]
    #
    indices = encoder.encode("low")
    assert(2 == len(indices))
    assert(encoder.lookup_token(indices[0]) == "lo")
    assert(encoder.lookup_token(indices[1]) == "w</w>")


#
# Test encode in combination with decode
#
def test12():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 3)
    #
    # check new vocab - we should have learned the rules w e ,  s t</w> and l o
    #
    _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'we', 'r</w>', 'n', 's', 't</w>', 'i', 'd', 'st</w>', 'lo','</unk>'}
    assert(_vocab == set(encoder.get_vocab()))
    #
    # Encode "st" --> should give st
    #
    indices = encoder.encode("low")
    #
    # Call decode on the result list of token - this should give the original word back
    #
    word = encoder.decode(indices)
    assert(word == "low</w>")

#
# Test encode in combination with decode, remove eow marker
#
def test13():
    pre_tokenized_text = ["low", "lower",  "newest", "widest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 3)
    #
    # check new vocab - we should have learned the rules w e ,  s t</w> and l o
    #
    _vocab = {'l', 'o', 'w</w>', 'w', 'e', 'we', 'r</w>', 'n', 's', 't</w>', 'i', 'd', 'st</w>', 'lo', '</unk>'}
    assert(_vocab == set(encoder.get_vocab()))
    #
    # Encode "st" --> should give st
    #
    indices = encoder.encode("st")
    assert(1 == len(indices))
    assert(encoder.lookup_token(indices[0]) == "st</w>")
    #
    # Encode low --> should give ["lo", "w</w>"]
    #
    indices = encoder.encode("low")
    assert(2 == len(indices))
    assert(encoder.lookup_token(indices[0]) == "lo")
    assert(encoder.lookup_token(indices[1]) == "w</w>")
    #
    # Call decode on the result list of token - this should give the original word back
    #
    word = encoder.decode(indices, remove_eow = True)
    assert(word == "low")

#
# Test encode for a full word that appears in the vocabulary
#
def test14():
    pre_tokenized_text = ["low", "lower",  "lowest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s  = 10)
    index_low = encoder["low</w>"]
    assert index_low is not None
    #
    # Encode word 
    #
    indices = encoder.encode("low")
    assert(1 == len(indices))
    assert(indices[0] == index_low)

#
# Test encode with unknown characters 

def test15():
    pre_tokenized_text = ["low", "lower",  "lowest"]
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s  = 2)
    #
    # Learns l o -> lo, w e -> we
    #
    indices = encoder.encode("allow")
    #
    # Should give <unk> l lo w
    #
    assert(4 == len(indices))
    assert(indices[0]==encoder.get_default_index())
    assert(indices[1]==encoder['l'])
    assert(indices[2]==encoder['lo'])
    assert(indices[3]==encoder['w</w>'])
    #
    # do once more and test caching
    #
    assert(encoder.is_cached("allow"))
    _indices = encoder.encode("allow")
    assert indices == _indices


#
# Save and load rules
#
def test16():
    pre_tokenized_text = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation".strip('\r\n ').split(' ')
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 8)
    assert(encoder.get_rules() == [
        ('s', 'i'),
        ('p', 'h'),
        ('t', 'h'),
        ('si', 'g'),
        ('s', 'e</w>'),
        ('r', 'a'),
        ('o', 'n</w>'),
        ('i', 't'),
    ])
    _, filename = tempfile.mkstemp()
    encoder.save_rules(filename)
    encoder.load_rules(filename)
    assert(encoder.get_rules() == [
        ('s', 'i'),
        ('p', 'h'),
        ('t', 'h'),
        ('si', 'g'),
        ('s', 'e</w>'),
        ('r', 'a'),
        ('o', 'n</w>'),
        ('i', 't'),
    ])
    os.remove(filename)

#
# Save and load rules and vocabulary
#
def test17():
    pre_tokenized_text = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation".strip('\r\n ').split(' ')
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 8)
    assert(encoder.get_rules() == [
        ('s', 'i'),
        ('p', 'h'),
        ('t', 'h'),
        ('si', 'g'),
        ('s', 'e</w>'),
        ('r', 'a'),
        ('o', 'n</w>'),
        ('i', 't'),
    ])
    _, vfile = tempfile.mkstemp()
    _, rfile = tempfile.mkstemp()
    encoder.save_rules(rfile)
    encoder.save_vocab(vfile)
    #
    # Build a second encoder 
    #
    _encoder = BPE.BPEEncoder()
    _encoder.load_rules(rfile)
    _encoder.load_vocab(vfile)
    assert(_encoder.get_rules() == encoder.get_rules())
    for idx, symbol in enumerate(encoder.get_vocab()):
        assert symbol == _encoder.lookup_token(idx), f"Symbols for index {idx} do not match"
        assert _encoder[symbol] == idx
    os.remove(vfile)
    os.remove(rfile)

#
# Test the full life cycle - train on a small text, encode a text, load and save vocabulary and rules, encode the same text 
# in a new encoder initialized from the files and compare
#
def test18():
    pre_tokenized_text = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation".strip('\r\n ').split(' ')
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 8)
    text = "Let us try encoding and decoding"
    encoded_text = [encoder.encode(word) for word in text.split()]
    _, vfile = tempfile.mkstemp()
    _, rfile = tempfile.mkstemp()
    encoder.save_rules(rfile)
    encoder.save_vocab(vfile)
    #
    # Build a second encoder 
    #
    _encoder = BPE.BPEEncoder()
    _encoder.load_rules(rfile)
    _encoder.compile_rules()
    _encoder.load_vocab(vfile)
    os.remove(vfile)
    os.remove(rfile)
    _encoded_text = [_encoder.encode(word) for word in text.split()]
    assert _encoded_text == encoded_text
    decoded_text = ""
    for word in encoded_text:
        decoded_text = decoded_text + _encoder.decode(word)
    decoded_text = re.sub(r"</w>", " ", decoded_text)
    assert decoded_text == "</unk>et us try encoding an</unk>decoding "


#
# Test alignment of vocabulary size to a multiple of 64
# 
def test19():
    pre_tokenized_text = "Python is a high-level, general-purpose programming language. Its design philosophy emphasizes code readability with the use of significant indentation".strip('\r\n ').split(' ')
    encoder = BPE.BPEEncoder()   
    encoder.init_vocab(pre_tokenized_text)
    encoder.learn(s = 8, align_vocab = True)
    assert(encoder.get_rules()[0:8] == [
        ('s', 'i'),
        ('p', 'h'),
        ('t', 'h'),
        ('si', 'g'),
        ('s', 'e</w>'),
        ('r', 'a'),
        ('o', 'n</w>'),
        ('i', 't'),
    ])
    #
    # Vocab size should be a multiple of 64. As our 
    # initial vocab size is 37 and we request 8 merges, the size
    # without the flag would be 37 + 8 = 45, so with alignment the vocab
    # size should be 64 and we expect 64 - 37 = 27 merges
    #
    assert len(encoder.get_rules()) == 27, "Number of merges not increased"


###################################################
#
# Run tests 
#
###################################################

parser = argparse.ArgumentParser();
parser.add_argument("--infile", type=str, default=None, help="Specify input file (if none run unit tests)")
parser.add_argument("--outfile", type=str, default=None, help="Output file for rules (only relevant if input file is given)")
parser.add_argument("--steps", type=int, default=50, help="Merges (only relevant if input file is given))")
args = parser.parse_args();

if args.infile is None:
    print("Running unit tests...")
    test1()
    test2()
    test3()
    test4()
    test5()
    test6()
    test7()
    test8()
    test9()
    test10()
    test11()
    test12()
    test13()
    test14()
    test15()
    test16()
    test17()
    test18()
    test19()
    print("OK")
else:
    print(f"Reading text from file {args.infile}")
    text = read_and_pretokenize(args.infile, progress_bar=True)
    print("Creating encoder")
    encoder = BPE.BPEEncoder(progress_bars=True)   
    print("Initializing vocabulary")
    encoder.init_vocab(text)
    print("Starting learning phase")
    encoder.learn(args.steps)
    if args.outfile:
        with open(args.outfile, "w") as f:
            for rule in encoder.get_rules():
                f.write(f"{rule[0]} {rule[1]}\n")
    


