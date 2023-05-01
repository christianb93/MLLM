import argparse
import os
import torch
import argparse
import time

import model
import util

#
# Training parameters
#
LR = 0.0005
DROPOUT = 0.2
BATCH_SIZE = 512


#
# Get parameters
#
parser = argparse.ArgumentParser();
parser.add_argument("--epochs", type=int, default=3, help="Number of epochs to train")
parser.add_argument("--batch_size", type=int, default=BATCH_SIZE, help="Batch size")
parser.add_argument("--log_steps", type=int, default=250, help="Number of steps after which we log progress")
parser.add_argument('--compile', action='store_true', default=False, help="Compile model")
parser.add_argument('--lr', type=float, default=LR, help="Learning rate")
parser.add_argument('--lr_min', type=float, default=LR*0.1, help="Minimal learning rate")
parser.add_argument("--dropout", type=float, default=DROPOUT, help="Dropout")
parser.add_argument("--autocast", action='store_true', default=False, help="Use autocast")
parser.add_argument("--model", type=str, default="model.pt", help="Name of the model to load as starting point")
parser.add_argument("--tf32", action='store_true', default=False, help="Enable TF32")


args = parser.parse_args();

train_ds = util.DataSet(window_size = model.CONTEXT_SIZE, encoder = None)
val_ds  = util.DataSet(window_size = model.CONTEXT_SIZE, data = "valid", encoder = train_ds.get_vocab())
print(f"Using data set with {len(train_ds)} training items and {len(val_ds)} validation items")


#
# Create data loader. We drop the last batch to avoid spikes in the loss function, and we use two workers 
# which should be enough to keep most GPUs busy
#
batch_size = args.batch_size
training_data = torch.utils.data.DataLoader(train_ds, batch_size = batch_size, collate_fn = util.collate_fn, shuffle = True, drop_last = True, num_workers = 2)
val_data = torch.utils.data.DataLoader(val_ds, batch_size = batch_size, collate_fn = util.collate_fn, shuffle = True, drop_last = True)

#
# Create model or load it if we had a copy on disk
#
device = "cuda" if torch.cuda.is_available() else "cpu"

#
# Enable TF32 if requested
#
if device != "cpu" and args.tf32:
    print("Enabling TF32 usage")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

V = len(train_ds.get_vocab())
_model = model.Model(vocab_size = V, model_dim = model.MODEL_DIM, context_size = model.CONTEXT_SIZE,  dropout = args.dropout).to(device)
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
if not args.compile:
    model_to_train = _model
else:
    print("Compiling model")
    model_to_train = torch.compile(_model)
model_to_train.train()
#
# Do actual training
#
print(f"Training for {args.epochs} epochs")
print(f"Using device {device} and batch size {batch_size}, lr = {args.lr} (min {args.lr_min}), dropout = {args.dropout}, autocast = {args.autocast}")
start_time = time.time()
losses, val_losses = util.train(model_to_train, args.epochs, batch_size = batch_size,
               train_data_loader = training_data, val_data_loader = val_data, 
               device = device, log_steps = args.log_steps, 
               lr = args.lr, autocast = args.autocast, lr_min = args.lr_min)
end_time = time.time()
print(f"Total training time: {end_time - start_time}")
#
# Save model
#
torch.save(_model.state_dict(), "model.pt")
print("Done saving model to model.pt, now writing losses")
#
# Write losses into a file in GNUPLOT format
#
f = open("losses.dat", "w")
f.write(f"# Total training time: {end_time - start_time}\n")
for epoch, loss in enumerate(val_losses):
    f.write(f"# Epoch {epoch} --> validation loss {loss}\n")    
f.write("plot '-' with lines linecolor 'blue'\n")
for i, l in enumerate(losses):
    f.write(f"{i}   {l}\n")
f.close()


