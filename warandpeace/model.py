#
# Our model
# We use an embedding layer, followed by an LSTM and finally an output layer
#
import torch 


#
# Default values for the parameter
#
# Inner dimension of the LSTM
D = 128
# Embedding dimension, i.e. input dimension of the LSTM
E = 32
# Number of layers
LAYERS = 4


class TheModel(torch.nn.Module):

    def __init__(self, vocab_size, hidden_dim = D, embedding_dim = E, num_layers = LAYERS, dropout = 0):
        super().__init__()
        self._embedding = torch.nn.Embedding(vocab_size, embedding_dim)
        self._lstm = torch.nn.LSTM(input_size = embedding_dim, hidden_size = hidden_dim, num_layers = num_layers, dropout = dropout)
        self._out = torch.nn.Linear(in_features = hidden_dim, out_features = vocab_size)

    def forward(self, x, hidden = None):
        x = self._embedding(x) # shape is now (L, B, E)
        x, hidden = self._lstm(x, hidden) # shape (L, B, H)
        x = self._out(x)
        return x, hidden

    
