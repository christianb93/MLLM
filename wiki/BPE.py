#
# Implementation of binary pair encoding 
#
# Sources:
# - the original paper "Neural Machine Translation of Rare Words with Subword Units" by R. Sennrich, B. Haddow and A. Birch, available at https://arxiv.org/abs/1508.07909
# - the reference implementation published by the first author at https://github.com/rsennrich/subword-nmt
#
# This is basically a simplified (but also less efficient) version of the reference implementation:
# - as the reference implementation, we use an index to keep track of the words in which a symbol pair appears
# - we update the index and the statistics incrementally after each merge, but use a simplified (though less efficient) approach to do this 
#
#
# Data structures:
#
# itos - a list of symbols (i.e. token) that we know
# stoi - a dictionary mapping a symbol to its index in itos
# rules - the list of rules that we have learned
# word_frequencies - a list of pairs (word, frequency), where each word is already split into a sequence of symbols
# stats - a dictionary indexed by symbol pairs that counts how often a symbol pair appears in the input
# index - a dictionary of dictionaries, index[a, b, word_idx] counts how often the symbol pair a, b appears in the word with position word_idx in word_frequencies 
#


import collections
import re

from tqdm import tqdm

class BPEEncoder:

    def __init__(self, progress_bars = False):
        self._word_frequencies = []
        self._itos = []
        self._stoi = dict()
        self._rules = []
        self._default_index = 0
        self._stats = collections.defaultdict(int)
        self._index = collections.defaultdict(dict)
        self._progress_bars = progress_bars
        self._cache = dict()
        self._pattern = []

    #
    # Add contribution of a single word to stats
    #
    def _add_stats_for_word(self, word_idx):
        word, freq = self._word_frequencies[word_idx]
        symbols = word.split()
        for i in range(len(symbols)-1):
            self._stats[symbols[i],symbols[i+1]] += freq
            current = self._index[symbols[i],symbols[i+1]].pop(word_idx, 0)
            self._index[symbols[i],symbols[i+1]][word_idx] = current + 1
            
    #
    # Update contribution of a single word to statistics and index
    #
    def _update_stats_for_word(self, word_idx, old_word, new_word, freq):
        #
        # Determine symbol pairs for both words, counting new symbol as +1
        # and old symbols as -1
        #
        pairs = collections.defaultdict(int)
        old_symbols = old_word.split()
        new_symbols = new_word.split()
        for i in range(len(old_symbols)-1):
            (a, b) = (old_symbols[i],old_symbols[i+1])
            ab = f"{a} {b}"
            pairs[ab] -= 1
        
        for i in range(len(new_symbols)-1):
            (a, b) = (new_symbols[i],new_symbols[i+1])
            ab = f"{a} {b}"
            pairs[ab] += 1
        #
        # Remove contributions of existing word and add contribution
        # of new word unless the contributions cancel
        #
        for ab, x in pairs.items():
            if x != 0:
                (a, b) = ab.split()
                self._stats[a,b] += x*freq
                count = self._index[a, b].pop(word_idx, 0) + x
                if count:
                    self._index[a, b][word_idx] = count
                
        #
        # Update word frequencies
        #
        self._word_frequencies[word_idx] = (new_word, freq)


    #
    # Build statistics from scratch
    #
    def _build_stats(self):
        for word_idx in range(len(self._word_frequencies)):
            self._add_stats_for_word(word_idx)


    def get_stats(self):
         return self._stats    

    #
    # Initialize vocabulary and rules - call this before
    # calling learn
    #
    def init_vocab(self, pre_tokenized_text):
        if self._progress_bars:
            print("Counting words in input")
        counter = collections.Counter(pre_tokenized_text)
        vocab = set()
        if self._progress_bars:
            print("Building word frequencies")

        _word_frequencies = {" ".join(word) + "</w>" : frequency for word, frequency in 
                                    tqdm(counter.items(), desc="Calculating frequencies", disable = not self._progress_bars) if len(word) > 0}
        self._word_frequencies = []
        for word, freq in _word_frequencies.items():
            self._word_frequencies.append((word, freq))
            for c in word.split():
                vocab.add(c)
        self._itos = ["</unk>", *vocab]
        for idx, symbol in enumerate(self._itos):
            self._stoi[symbol] = idx
        self._rules = []
        self._build_stats()

    #
    # Get vocabulary
    # 
    def get_vocab(self):
         return self._itos
    
    #
    # Do a single merge and incrementally update statistics
    #
    def _do_merge(self, best_pair):
        word_indices_to_visit = [word_idx for word_idx, count in self._index[best_pair[0], best_pair[1]].items()]
        new_token = "".join(best_pair)
        #
        # Make sure to merge only freestanding token. And do not forget to escape
        # the pair, as it might contain for instance dots
        #
        pattern = re.compile(r'(?<!\S)' + re.escape(" ".join(best_pair)) + r'(?!\S)')
        self._itos.append(new_token)
        self._stoi[new_token] = len(self._itos) - 1
        for word_idx in word_indices_to_visit:
            word, freq = self._word_frequencies[word_idx]
            new_word = pattern.sub(new_token, word)
            #
            # Update statistics - remove contribution of old word and add contribution of new word
            #
            if word != new_word:
                self._update_stats_for_word(word_idx, word, new_word, freq)
            
            
    #
    # Do s learning iterations, i.e. determine most frequent pair, do a merge and extend the vocabulary
    #
    def learn(self, s = 1, min_frequency = 1, align_vocab = False):
        #
        # If requested make sure that the size of the vocabulary is a multiple of 64
           #   
        if align_vocab:
            vocab_size = len(self._itos)
            predicted_size = s + vocab_size
            if (predicted_size % 64):
                predicted_size = (predicted_size // 64 + 1) * 64
                s = predicted_size - vocab_size
        for i in tqdm(range(s), desc = "Merging", disable = not self._progress_bars):
            best_pair = max(self._stats, key=lambda x: (self._stats[x], x)) 
            if (self._stats[best_pair] < min_frequency):
                break
            self._do_merge(best_pair)
            self._rules.append(best_pair)

    #
    # Get the learned rules
    #     
    def get_rules(self):
        return self._rules

    #
    # Return an index for a token
    #
    def __getitem__(self, symbol):
        if symbol in self._itos:
            return self._stoi[symbol]
        else:
            return self._default_index
    
    def __len__(self):
        return len(self._itos)
    
    #
    # Return a token for an index
    #
    def lookup_token(self, idx):
        return self._itos[idx]
    
    #
    # Get index of </unk>
    #
    def get_default_index(self):
        return self._default_index
    

    #
    # Encode a word by applying all learned rules
    #
    def encode(self, word):
        #
        # Lookup in cache
        #
        if word in self._cache:
            return self._cache[word]
        _word = " ".join(word) + "</w>"
        #
        # Shortcut - lookup word itself
        #
        if _word in self._itos:
            return [self._stoi[_word]]
        #
        # apply rules in the original order
        #
        for r, bp in enumerate(self._rules):
            new_token = "".join(bp)
            if self._pattern:
                pattern = self._pattern[r]  
            else:
                pattern = re.compile(r'(?<!\S)' + re.escape(" ".join(bp)) + r'(?!\S)')
            _word = pattern.sub(new_token, _word)
            #
            # Early stopping if word does not contain any more spaces
            #
            if -1 == _word.find(" "):
                break

        #
        # Translate into indices
        #
        indices = [self.__getitem__(symbol) for symbol in _word.split()]
        self._cache[word] = indices
        return indices
    
    #
    # Decode, i.e. turn a list of indices into a word. Note that the output still contains the end-of-word symbol
    #
    def decode(self, tokens, remove_eow = False):
        word = ""
        for idx in tokens:
            word = word + self._itos[idx]
        if remove_eow:
            word = re.sub(r"</w>", "", word)
        return word
    
    #
    # Write rules to disk
    #
    def save_rules(self, filename):
        with open(filename, "w") as f:
            for (a, b) in self._rules:
                f.write(f"{a} {b}\n")

    #
    # Load rules from disk
    #
    def load_rules(self, filename):
        with open(filename, "r") as f:
            self._rules = [tuple(line.split()) for line in f]

    #
    # Write vocabulary to disk, i.e. the list of symbols. We write one symbol per line
    #
    def save_vocab(self, filename):
        with open(filename, "w") as f:
            for symbol in self._itos:
                f.write(f"{symbol}\n")

    #
    # Load symbols from disk
    #
    def load_vocab(self, filename):
        with open(filename, "r") as f:
            self._itos = [line.strip("\t\n ") for line in f]
        for idx, symbol in enumerate(self._itos):
            self._stoi[symbol] = idx
        self._cache = dict()

    #
    # Compile rules
    #
    def compile_rules(self):
        self._pattern = []
        for bp in self._rules:
            pattern = re.compile(r'(?<!\S)' + re.escape(" ".join(bp)) + r'(?!\S)')
            self._pattern.append(pattern)

    #
    # Check whether a word is in the cache 
    #
    def is_cached(self, word):
        return word in self._cache
