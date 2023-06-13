import torch
import transformers 

models = {
    0 : "microsoft/dialogpt-small",
    1 : "microsoft/dialogpt-medium",
    2 : "microsoft/dialogpt-large"
}

def show_models():
    """
    Print all available models
    """
    for model_id, model in models.items():
        print(f"Model ID: {model_id} ---> {model}")

def cuda_available():
    """
    Is CUDA available?
    """
    return torch.cuda.is_available()

def get_device():
    """
    Determine which devices we have available

    Returns:
        "cuda" if a CUDA-capable GPU is found, "cpu" otherwise
    """
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"

def get_model_name(model_id):
    """
    Return the name of a known model

    Args:
        model_id: the ID of the model

    Returns: the model name as a string
    """
    if model_id in models.keys():
        return models[model_id]
    else:
        raise ValueError("fUnknown model ID {model_id}")

def get_model_and_tokenizer(model_id):
    model_name = get_model_name(model_id)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_name)
    model = transformers.AutoModelForCausalLM.from_pretrained(model_name)
    return model, tokenizer


def do_p_sampling(p, p_val = 0.95, greedy = False):
    """
    Apply nucleus sampling to a distribution p

    Args:
        p: the distribution as a one-dimensional tensor
        p_val: the p-value used for nucleus sampling
        greedy: use greedy sampling instead of nucleus sampling

    Returns:
        the index that has been sampled (as ordinary number)
    """
    assert 1 == len(p.shape), "Input is assumed to be a 1-D tensor"
    if greedy:
        return torch.argmax(p).item()
    items , indices = torch.sort(p, descending = True)    
    _k = max((torch.cumsum(items, dim = 0) <= p_val).to(int).sum().item(), 1)
    keep = indices[:_k]
    _p = [p[i] for i in keep]
    idx = torch.distributions.categorical.Categorical(probs = torch.tensor(_p)).sample().item()
    idx = keep[idx]
    return idx.item()


def generate(model, tokenizer, input_ids, past_key_values = None, length = 50, p_val = 0.95, temperature = 0.7, debug = True, skip_eos = False, greedy = True):
    """
    Sample a response for a set of token IDs. We sample new token from the model until we either hit upon an end-of-sentence token
    or have reached a certain maximum length
    
    This function accepts an additional parameter `past_key_values` to cache keys and values in the attention layer for positions in the past, i.e. 
    processed during a preivous call. 
    
    Note that - this is a difference to the semantic on the model level - the input_ids  should still contain the full history, including those covered by 
    past_key_values, the overlap will be adjusted automatically before calling the actual model. 

    Args:
        model: the model to sample from
        tokenizer: the tokenizer
        input_ids: a list of token IDs that make up the history of the conversation so far (should end with <eos>)
        past_key_values: cached keys and values. Pass this to all calls but the first one using the returned value
        length: the number of token that we sample at most
        p_val: p-value to use for sampling
        temperature: temperature to use
        debug: trigger debugging output
        skip_eos: do not stop generation if we hit upon eos
        greedy: use greedy sampling

    Returns: 
        a tuple consisting of a sequence of generated token IDs, the `past_key_values` to be used for the next call and an array containing 
        the logits for each sampled token (i.e. each entry in the array is a tensor of shape V)
    """
    all_logits = []
    device = next(model.parameters()).device
    generated = []
    with torch.no_grad():
        if past_key_values is None:
            if debug:
                print(f"Using all input_ids {input_ids}")
            #
            # If there are no past keys and values, feed full prompt into model
            #            
            _input_ids = torch.tensor(input_ids, dtype=torch.long).unsqueeze(dim = 0)
            out = model(input_ids = _input_ids.to(device))
            logits = out.logits[:, -1, :] # shape B x V
        else:
            #
            # We have past keys and values. Only feed those items in the input ids which
            # are not covered by that, i.e. we pass
            # input_ids[past_keys_len]
            # where past_keys_len is the length of the cached keys
            #
            # past_key_values has shape B x H x L x head_dim, so we can get this length from 
            # the third dimension
            #
            past_keys_len = (past_key_values[0][0]).shape[2]
            if debug:
                print(f"Using input ids {input_ids[past_keys_len:]}")
            assert len(input_ids) > past_keys_len,  (
                                                        f"Input IDs have only {len(input_ids)} elements" 
                                                        f"which is less than the length"
                                                        f" of the past keys and values {len(past_key_values)}"
                                                    )
            _input_ids = torch.tensor(input_ids[past_keys_len:], dtype=torch.long).unsqueeze(dim = 0)
            out = model(input_ids = _input_ids.to(device), 
                        past_key_values = past_key_values)
            logits = out.logits
            if len(logits.shape) == 3:
                logits = logits[:, -1, :] # reduce to shape B x V
        #
        # At this points, logits will have shape B x V. Remove batch dimension
        #
        assert 2 == len(logits.shape), "Logits should have shape B x V"
        logits = logits.squeeze(dim = 0)
        past_key_values = out.past_key_values
        while (len(generated) < length):
            #
            # Sample new index and append to encoded sample
            #
            all_logits.append(logits)
            p = torch.nn.functional.softmax(logits / temperature, dim = -1)
            idx = do_p_sampling(p, p_val, greedy = greedy)
            generated.append(idx)
            #
            # Feed new index as input_ids and cached keys and values as past_key_values
            #
            out = model(input_ids = torch.tensor(idx, dtype=torch.long).unsqueeze(dim = 0).to(device), past_key_values = past_key_values)
            logits = out.logits
            assert 2 == len(logits.shape), "Logits should have shape B x V"
            logits = logits.squeeze(dim = 0)
            past_key_values = out.past_key_values
            if debug:
                print(f"Generated token: {generated}")
                print(tokenizer.decode(generated))
            #
            # If this is an end-of-sentence marker stop generating. Note that we still need the forward pass to get the full past_key_values 
            # including that for the position of the eos
            #
            if idx == tokenizer.eos_token_id and not skip_eos:
                break
        
        assert len(input_ids) + len(generated) == (past_key_values[0][0]).shape[2], "Key values do not seem to have the correct length"
        return generated, past_key_values, all_logits
