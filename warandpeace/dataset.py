import torch 
import json

#
# A dataset for our training. We load data fron either train.json
# or val.json
#

class BookDataset(torch.utils.data.Dataset):

    def __init__(self, window_size = 64, data = "train", limit = None):
        super().__init__()
        #
        # Load data
        #
        if data == "train":
            filename = "train.json"
        else:
            filename = "val.json"
        with open(filename, "r") as f:
            self._encoded_book = json.load(f)
        self._window_size = window_size
        #
        # For index i, we need window_size + i + 1 still to be a valid 
        # index into the encoded book, i.e. the index we can serve is constrained by
        # i  < len(book) - window_size - 1
        # so that the length of our dataset is len(book) - window_size - 1
        if limit is None:
            self._len = len(self._encoded_book) - window_size - 1
        else:
            self._len = limit
        assert self._len > 0, "Book is to short, please validate input file"
 
    def __getitem__(self, i):
        if (i < self._len):
            x = self._encoded_book[i: i + self._window_size]
            y = self._encoded_book[i + 1: i + self._window_size + 1]
            return torch.tensor(x), torch.tensor(y)
        else:
            raise KeyError

    def __len__(self):
        return self._len
    

