import torch
import os
import argparse 
import re

import model
import util
import BPE

VFILE= "vocab.dat"
RFILE = "rules.dat"

#
# Get parameters
#
parser = argparse.ArgumentParser();
parser.add_argument("--method", type=int, default = 3)
parser.add_argument("--length", type=int, default = 200, help = "Length of sample to create")
parser.add_argument("--model", type=str, default = "model.pt", help = "Name of the model to load as starting point")
parser.add_argument("--cpu", action="store_true", default = False, help = "Force usage of CPU even if GPU is available")
parser.add_argument("--prompt", type=str, default = ". ", help = "Prompt to use")
parser.add_argument("--temperature", type=float, default = 0.7, help = "Temperature to use for sampling")


args = parser.parse_args();


device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
print(f"Using device {device} for inference")

#
# Load encoder and model
#
encoder = BPE.BPEEncoder(progress_bars = True)
encoder.load_rules(RFILE)
encoder.load_vocab(VFILE)
vocab = encoder.get_vocab()
V = len(vocab)
_model = model.Model(vocab_size = V).to(device)
print(f"Checking for existing model")
if os.path.exists(args.model):
    print("Loading model state from disk")
    _model.load_state_dict(torch.load(args.model, map_location = device))
    _model.to(device)
else:
    print(f"Could not find model {args.model}")
    exit(1)
_model.eval()



sample_ids = util.do_sample(args.prompt, _model, encoder, device = device, method = args.method, length = args.length, temperature = args.temperature)
sample = util.beautify_decoder_output(encoder.decode(sample_ids))


#
# Strip off prompt again if we have used the default prompt
#
if args.prompt == ". ":
    sample = sample[len(args.prompt):]
print(sample)
