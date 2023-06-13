#
# Use DialoGPT from the Huggingface hub for a simple chatbot
#
import argparse

import transformers
import torch


import utils

    

#####################################################
# Main
#####################################################

parser = argparse.ArgumentParser();
parser.add_argument("--show-models", action="store_true", default = False, help = "Show list of supported models")
parser.add_argument("--model", type = int, default = 2, help = "Model ID - use --show-models to get a list of options")
parser.add_argument("--token", type = int, default = 30, help = "Number of token to be returned (including prompt)")
parser.add_argument("--device", type = str, default = "cpu", help = "Device to use (cuda or cpu)")
parser.add_argument("--temperature", type = float, default = 0.7, help = "Temperature used for sampling")
parser.add_argument("--p_val", type = float, default = 0.95, help = "Parameter for top-p sampling")
parser.add_argument("--debug", action = "store_true", default = False, help = "Print intermediate results")

args = parser.parse_args()

#
# Print list of models and exit if requested
#
if args.show_models:
    utils.show_models()
    exit(0)


model_name = utils.get_model_name(args.model)
#
# Get tokenizer and model
#
print(f"Getting tokenizer and weights for model {model_name}")
model, tokenizer = utils.get_model_and_tokenizer(args.model)
model.eval()
assert model.config.use_cache, "Caching of keys and values not enabled, what went wrong?"
#
# Determine device to use
#
if args.device == "cpu":
    device = "cpu"
else:
    device = utils.get_device()
print(f"Running inference on device {device}")
model = model.to(device)

#
# Do the conversation. In each turn of the conversation, we do the following:
# - get a new prompt from the user
# - encode the prompt and add an EOS marker
# - concatenate that with the full conversation so far (which therefore consists of user and bot text, separated by EOS markers)
# - feed that as prompt to the generator 
# - append the result as well to the conversation
#
# So at the each point the conversation looks as follows:
# <Prompt of user><EOS><Bot output><EOS><Prompt of user><EOS><Bot output><EOS>
#
#
input_ids = []
past_key_values = None
while True:
    prompt = input(">>> You: ")
    if "" == prompt:
        exit(0)
    input_ids.extend(tokenizer.encode(prompt))
    input_ids.append(tokenizer.eos_token_id)
    if args.debug:
        print(f"Passing input_ids {input_ids} to generate function")
    generated, past_key_values, _ = utils.generate(model = model, 
                                            tokenizer = tokenizer, 
                                            input_ids = input_ids, 
                                            past_key_values = past_key_values, 
                                            length = args.token, 
                                            temperature = args.temperature, 
                                            p_val = args.p_val, 
                                            debug = args.debug)
    print(f">>> Bot: {tokenizer.decode(generated).replace('<|endoftext|>', '')}")
    #
    # Next turn
    #
    input_ids.extend(generated)
