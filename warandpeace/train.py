import argparse
import os
import torch
import time 
import argparse
import tqdm

import model
import dataset


#
# Training parameters
#
LR = 0.001
DROPOUT = 0.2
WINDOW_SIZE = 64
EPOCHS = 7
BATCH_SIZE=1024

#
# Collate function for batch loading
# We use the second dimension as the batch dimension
# so that the output has shape L x B
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
# A simple training loop
#
def train(model, epochs, train_data_loader, val_data_loader, lr = LR, device = "cpu", log_steps = 250):
    losses = []
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    loss_fn = torch.nn.functional.cross_entropy
    steps = 0    
    start_logging_period = time.time()
    model.train()
        
    for epoch in range(epochs):
        start_epoch = time.time()
        for x, y in tqdm.tqdm(train_data_loader,desc = f"Epoch {epoch}"):
            steps += 1
            optimizer.zero_grad()

            f, _ = model(x.to(device))
            V = f.shape[2]
            f = f.view(-1, V)            
            y = y.to(device).flatten()
            loss = loss_fn(f, y)
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (0 == (steps % log_steps)):
                time_per_step = (time.time() - start_logging_period) / log_steps
                start_logging_period = time.time()
                print(f"Completed {steps} steps ({time_per_step:.3f} seconds per step), current loss is {loss.item()}")

        elapsed_time = time.time() - start_epoch
        #
        # Do validation
        #
        val_loss = 0
        items_in_val = 0
        if val_data_loader is not None:
            model.eval()
            with torch.no_grad():
                for x, y in val_data_loader:
                    f, _ = model(x.to(device))
                    f = f.view(-1, V)            
                    y = y.to(device).flatten()
                    val_loss = val_loss + loss_fn(f, y).item()
                    items_in_val += 1
            val_loss = val_loss / items_in_val
            model.train()
        print(f"Completed epoch {epoch}, validation loss is {val_loss}, duration {elapsed_time} seconds, {steps} done in total")
        #
        # Save checkpoint
        #
        chkpt_name = f"model_{epoch}.pt"
        torch.save(_model.state_dict(), chkpt_name)
        print(f"Saved model checkpoint at {chkpt_name}")
    
    return losses


#
# Get parameters
#
parser = argparse.ArgumentParser();
parser.add_argument("--epochs", type=int, default=EPOCHS)
parser.add_argument("--batch_size", type=int, default=1024)
parser.add_argument("--log_steps", type=int, default=250)
parser.add_argument('--lr', type=float, default=LR)
parser.add_argument("--dropout", type=float, default=DROPOUT)
parser.add_argument("--model", type=str, default="model.pt", help="Name of the model to load as starting point")
parser.add_argument("--limit", type=int, default=None, help="Limit the dataset for testing purposes")


args = parser.parse_args();

train_ds = dataset.BookDataset(window_size = WINDOW_SIZE, limit = args.limit)
val_ds  = dataset.BookDataset(window_size = WINDOW_SIZE, data = "val", limit = int(args.limit * 0.1) if args.limit is not None else None)
print(f"Using data set with {len(train_ds)} training items and {len(val_ds)} validation items")


#
# Create data loader. We drop the last batch to avoid spikes in the loss function
#
batch_size = args.batch_size
training_data = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, 
                                            collate_fn = collate_fn, 
                                            shuffle = True, 
                                            drop_last = True, 
                                            num_workers = 2)
val_data = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, 
                                            collate_fn = collate_fn, 
                                            shuffle = True, 
                                            drop_last = True)

#
# Create model or load it if we had a copy on disk
#
device = "cuda" if torch.cuda.is_available() else "cpu"

#
# Load vocab
#
vocab = torch.load("vocab.pt")
V = len(vocab)

_model = model.TheModel(vocab_size = V,  dropout = args.dropout).to(device)
print(f"Checking for existing model {args.model}")
if os.path.exists(args.model):
    print("Loading model state from disk")
    _model.load_state_dict(torch.load(args.model))
    _model.to(device)
else:
    print("Starting over with a new model")
_model.train()    
#
# Do training. If requested, we compile the model (only works with PyTorch >= 2.0)
#
print(_model)
losses = []
steps_per_epoch = train_ds.__len__() // batch_size
print(f"Training for {args.epochs} epochs ({steps_per_epoch} steps per epoch), using device {device} and batch size {batch_size}, lr = {args.lr}")
start_time = time.time()
losses = train(_model, 
               epochs = args.epochs, 
               train_data_loader = training_data,
               val_data_loader = val_data, 
               device = device, 
               log_steps = args.log_steps, 
               lr = args.lr)
  
#
# Save model
#
torch.save(_model.state_dict(), "model.pt")
print("Done saving model to model.pt, now writing losses")
#
# Save losses
#
f = open("losses.dat", "w")
f.write(f"# Epochs =  {args.epochs}, batch size =  {args.batch_size}, learning rate =  {args.lr}, training on device {device}\n")
f.write(f"# Total training time: {(time.time() - start_time) / 60} minutes\n")
if device == "cuda":
    f.write(f"# CUDA device: {torch.cuda.get_device_name(device)} - {torch.cuda.get_device_properties(device)}\n")
f.write("plot '-' with lines linecolor 'blue'\n")
for i, l in enumerate(losses):
    f.write(f"{i}   {l}\n")
f.close()


