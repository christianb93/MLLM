
import re
import torch

#
# A simple tokenizer that converts the input into a
# sequence of characters
#
def tokenize(text):
    #
    # Remove all unknown characters
    # 
    text = re.sub(r"[^A-Za-z0-9 \-\.;,\n\?!]", '', text)
    #
    # Replace linebreaks and multiple spaces by a single space
    #
    text = re.sub("\n+", " ", text)
    text = re.sub(" +", " ", text)
    #
    # Split into characters
    #
    return [_t for _t in text]

#
# Decode a sequence of ids
#
def decode(indices, vocab):
    return "".join(vocab.lookup_tokens(indices))


#
# Sample starting with a given prompt 
# Supported methods:
# 0 - greedy search
# 1 - multinomial sampling
# 2 - top-k sampling
# 3 - top-p (nucleus) sampling
#
@torch.no_grad()
def do_sample(prompt, model, vocab, length = 50, device = "cpu", temperature = 0.7, method = 3, k_val = 5, p_val = 0.95):
    input_ids = [vocab[t] for t in tokenize(prompt)]
    hidden = None
    while (len(input_ids) < length):
        x = torch.tensor(input_ids, dtype = torch.long).to(device)
        #
        # Feed input ids into model
        #
        if hidden is None:
            f, hidden = model(x)
        else:
            f, hidden = model(x[-1].unsqueeze(dim = 0), hidden)
    	#
        # f has shape (L, V). Take last element and apply softmax as well as temperature
        #
        p = torch.softmax(f[-1] / temperature, dim = 0)
        #
        # Sample
        if 0 == method:
            idx = torch.argmax(p).item()
        elif 1 == method:
            idx = torch.distributions.categorical.Categorical(probs = p).sample().item()
        elif 2 == method:
            #
            # Sort and remove all indices after the k-th index
            #
            _, indices = torch.sort(p, descending = True)
            keep = indices[:k_val]
            #
            # Sample over the items that are left
            #
            _p = [p[i] for i in keep]
            idx = torch.distributions.categorical.Categorical(probs = torch.tensor(_p)).sample().item()
            idx = keep[idx]        
        elif 3 == method:    
            items , indices = torch.sort(p, descending = True)    
            items = iter(items.tolist())
            _sum = 0
            _k = 0
            while _sum <= p_val:
                _sum, _k =  _sum + next(items), _k + 1
            keep = indices[:_k]
            _p = [p[i] for i in keep]
            idx = torch.distributions.categorical.Categorical(probs = torch.tensor(_p)).sample().item()
            idx = keep[idx]    
        else: 
            print(f"Sampling method {method} not supported")
            exit(1)
        input_ids.append(idx)

    return decode(input_ids, vocab)
