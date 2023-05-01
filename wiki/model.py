import torch

#
# Parameters for the model
#
CONTEXT_SIZE = 48
MODEL_DIM = 192
FF_DIM = 1024
HEADS = 8
LAYERS = 4
DROPOUT = 0.2


#
# A layer that adds a positional embedding to a given input. We use a fixed
# sinusoidal embeddings as in the original "Attention is all you need" paper
#
class PosEmbeddingLayer(torch.nn.Module):
    
    def __init__(self, context_size, model_dim):
        super().__init__()
        _p = torch.arange(context_size , dtype=torch.float32).unsqueeze(dim = 1)
        _i = torch.arange(0, model_dim , step = 2) / model_dim 
        x = _p / torch.pow(10000, _i)
        self.register_buffer("_pe", torch.zeros(context_size, model_dim))        
        self._pe[:, 0::2] = torch.sin(x)
        self._pe[:, 1::2] = torch.cos(x)
        self._pe = torch.unsqueeze(self._pe, dim = 1)
        self._model_dim = model_dim
        self._context_size = context_size
        
    #
    # Shape of input: (sequence length, batch_size, model_dim) 
    #
    def forward(self, x):
        assert len(x.shape) == 3, "Shape (L, B, D) required"
        assert x.shape[2] == self._model_dim, "Last dimension needs to be model dimension"
        L = x.shape[0]
        assert L <= self._context_size, "Length exceeds context size"
        pe = self._pe[:L, :, :]
        return x + pe
    
#
# A simple decoder-only transformer model, with a learned
# word embedding, a sinusoidal positional embedding and a linear
# output layer
#
class Model(torch.nn.Module):
    
    def __init__(self, vocab_size, model_dim = MODEL_DIM, context_size = CONTEXT_SIZE, ff_dim = FF_DIM, heads = HEADS, layers = LAYERS, dropout = DROPOUT):
        super().__init__()
        self._word_embedding = torch.nn.Embedding(vocab_size, model_dim)
        self._pe_embedding = PosEmbeddingLayer(context_size, model_dim)
        layer = torch.nn.TransformerEncoderLayer(d_model = model_dim, nhead = heads, dim_feedforward = ff_dim, dropout = dropout)
        self._transformer = torch.nn.TransformerEncoder(layer, num_layers = layers)
        self._linear = torch.nn.Linear(in_features = model_dim, out_features = vocab_size)
        self._model_dim = model_dim
        self._context_size = context_size
        self._vocab_size = vocab_size
        cached_mask = torch.tril(torch.ones(context_size, context_size)*(-1)*float('inf'), diagonal = -1).t()
        self.register_buffer("_cached_mask", cached_mask)
    
    #
    # Create a causal self-attention mask
    #
    def get_self_attention_mask(self):
        return self._cached_mask
        
    #
    # Shape of input: (L, B)
    # 
    def forward(self, x):
        assert len(x.shape) == 2, "Expecting two-dimensional input"
        (L, B) = x.shape
        x = self._word_embedding(x) # shape (L, B, model_dim)
        x = self._pe_embedding(x) 
        #
        # Mask input. As we have registered this is a buffer, it
        # should already be on the same device as the model
        #
        mask = self.get_self_attention_mask()[:L, :L]
        x = self._transformer(x, mask = mask)
        return self._linear(x)        

    #
    # Get context size
    #
    def get_context_size(self):
        return self._context_size

