import torch
import math
import re
import json
import BPE
import time
import tqdm

#
# A simple dataset for use with a PyTorch data loader
# It expects the actual data (encoded, in JSON format) in train.json
# or val.json, depending on the value of the data parameter.
#
# The getitem method returns a pair consisting of input and labels,
# using the usual teacher forcing to create the labels from the input
#
class DataSet(torch.utils.data.Dataset):
    
    def __init__(self, window_size = 8, data = "train", encoder = None):
        super().__init__()
        #
        # Load data
        #
        if data == "train":
            filename = "train.json"
        else:
            filename = "val.json"
        with open(filename, "r") as f:
            self._encoded_token = json.load(f)
        if encoder is None:
            self._encoder = BPE.BPEEncoder()
            self._encoder.load_vocab("vocab.dat")
            self._encoder.load_rules("rules.dat")
            self._encoder.compile_rules()
        else:
            self._encoder = encoder
        self._window_size = window_size
        self._len = len(self._encoded_token) - window_size - 1
        self._V = len(self._encoder)
        
    def get_vocab(self):
        return self._encoder
        
    def __getitem__(self, i):
        if (i < self._len):
            x = self._encoded_token[i: i + self._window_size]
            y = self._encoded_token[i + 1: i + self._window_size + 1]
            _x = torch.tensor(x)
            _y = torch.tensor(y)
            return _x, _y
        else:
            raise KeyError

    def __len__(self):
        return self._len
    

#
# Return a simple tokenizer 
#
def get_tokenizer():
    return lambda x : x.strip("\r\n ").split()


#
# Sample starting with a given prompt 
# Supported methods:
# 0 - greedy search
# 1 - multinomial sampling
# 2 - top-k sampling
# 3 - top-p (nucleus) sampling
#
@torch.no_grad()
def do_sample(prompt, model, encoder, length = 50, device = "cpu", temperature = 0.7, method = 3, k_val = 5, p_val = 0.95, pre_tokenizer = None):
    context_size = model.get_context_size()
    #
    # Pre-tokenize prompt and encode
    #
    if pre_tokenizer is None:
        pre_tokenizer = get_tokenizer()
    pre_tokenized_prompt = pre_tokenizer(prompt)
    input_ids = []
    for word in pre_tokenized_prompt:
        input_ids.extend(encoder.encode(word))
    #
    # Sample next word until we have reached the desired length
    # 
    while (len(input_ids) < length):
        #
        # Convert to tensor and add batch dimension
        #
        x = torch.tensor(input_ids[-context_size:], dtype = torch.long).to(device).unsqueeze(dim = 1)
        #
        # Feed input ids into model, get output and remove batch dimension again
        #
        f = model(x)
        f = f[:, 0, :]
    	#
        # f now has shape (L, V). Take last element and apply softmax as well as temperature
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
            #
            # Sum up until we reach p_val total probability mass
            #
            while _sum <= p_val:
                _sum, _k =  _sum + next(items), _k + 1
            #
            # now use this value of k 
            #
            keep = indices[:_k]
            _p = [p[i] for i in keep]
            idx = torch.distributions.categorical.Categorical(probs = torch.tensor(_p)).sample().item()
            idx = keep[idx]    
        else: 
            print(f"Sampling method {method} not supported")
            exit(1)
        input_ids.append(idx)

    return input_ids


#
# Beautify output of the BPE decoder a bit
#
def beautify_decoder_output(text):
    #
    # Replace end-of-word marker by a space
    #
    text = re.sub("</w>", " ", text)
    #
    # Remove spaces preceeding a punctuation mark
    #
    text = re.sub(r" ([\.,!\?;])", "\g<1>", text)
    return text



#
# Collate function for batch loading
#
def collate_fn(list):
    batch_size = len(list)
    _x = []
    _y = []
    for i in list:
        x,y = i
        _x.append(x)
        _y.append(y)
    return torch.stack(_x, dim = 1), torch.stack(_y, dim = 1)

#
# A cosine annealer with warm-up
#
class WarmupCosineAnnealingLR(torch.optim.lr_scheduler._LRScheduler): # use _LRScheduler to make it work with versions < 2.0
    
    def __init__(self, optimizer, T_max, warmup = 2000, lr_max = 0.00025, lr_min = None):
        self._t_max = T_max
        self._lr_max = lr_max
        self._optimizer = optimizer
        self._warmup = warmup
        if lr_min is None:
            self._lr_min = self._lr_max / 10
        else:
            self._lr_min = lr_min
        # Call this after initialization as the superclass will call get_lr
        super().__init__(optimizer)
        
    def get_lr(self):
        #
        # Return an array of learning rates per parameter group
        #
        w = self._warmup
        t = self._step_count
        if t < w:
            return [ float(t) / w * self._lr_max for param_group in self._optimizer.param_groups]
        else:
            return [ self._lr_min + 0.5*(self._lr_max - self._lr_min)*(1 + math.cos((t - w)/(self._t_max - w)*math.pi)) for param_group in self._optimizer.param_groups]    


#
# Do a single training step
# 
def train_step(model, x, y, optimizer, scheduler, scaler, autocast = False, loss_fn = torch.nn.functional.cross_entropy, device = "cpu"):
    optimizer.zero_grad()

    if autocast:
        with torch.autocast("cuda", dtype=torch.float16):
            f = model(x.to(device))
            V = f.shape[2]
            f = f.view(-1, V)            
            y = y.to(device).flatten()
            loss = loss_fn(f, y)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
    else:
        f = model(x.to(device))
        V = f.shape[2]
        f = f.view(-1, V)            
        y = y.to(device).flatten()
        loss = loss_fn(f, y)
        loss.backward()
        optimizer.step()

    scheduler.step()
    return loss.item()

#
# Evaluate on the validation data
#
def eval_model(model, val_data_loader, loss_fn = torch.nn.functional.cross_entropy,  device = "cpu"):
    val_loss = 0
    items_in_val = 0
    model.eval()
    with torch.no_grad():
        for x, y in val_data_loader:
            f = model(x.to(device))
            V = f.shape[2]
            f = f.view(-1, V)            
            y = y.to(device).flatten()
            loss = loss_fn(f, y)
            val_loss = val_loss + loss.item()
            items_in_val = items_in_val + 1
    val_loss = val_loss / items_in_val
    model.train()
    return val_loss


#
# The training loop
# 
def train(model, epochs, train_data_loader, val_data_loader, lr = 0.00025, device = "cpu",  lr_min = None, log_steps = 250, autocast = False, batch_size = None):
    #
    # Initialize some statistics
    #
    steps_per_epoch = train_data_loader.dataset.__len__() // batch_size
    print(f"Steps per epoch: {steps_per_epoch}")
    losses = []
    val_losses = []
    steps = 0    
    start_logging_period = time.time()
    if lr_min is None:
        lr_min = 0.1*lr
    #
    # Get optimizer, scheduler and scaler
    #
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = WarmupCosineAnnealingLR(optimizer, T_max = steps_per_epoch * epochs, lr_max = lr, warmup = 100, lr_min = lr_min)
    scaler = torch.cuda.amp.GradScaler()
    if device == "cpu":
        autocast = False
    #
    # Train 
    #
    for epoch in range(epochs):
        epoch_loss = 0
        start_epoch = time.time()
        for x, y in tqdm.tqdm(train_data_loader, desc=f"Epoch {epoch}"):
            steps += 1
            loss = train_step(model, x, y, optimizer, scheduler, scaler, autocast, device = device)
            losses.append(loss)
            epoch_loss = epoch_loss + loss
            if (0 == (steps % log_steps)):
                time_per_step = (time.time() - start_logging_period) / log_steps
                start_logging_period = time.time()
                print(f"Completed {steps} steps ({time_per_step:.3f} seconds per step), current loss is {loss}")
        #
        # Do validation
        #
        val_loss = eval_model(model, val_data_loader, device = device)
        val_losses.append(val_loss)
        #
        # Print summary and save checkpoint
        #
        print(f"Completed epoch {epoch}, validation loss is {val_loss:.4f}, duration {(time.time() - start_epoch):.2f} seconds, current learning rate is {optimizer.param_groups[0]['lr']:.6f}")
        chkpt_name = f"model_{epoch}.pt"
        torch.save(model.state_dict(), chkpt_name)
        print(f"Saved model checkpoint at {chkpt_name}")
    
    return losses, val_losses